#!/usr/bin/env python3
"""Cross-session stimulation-target transfer, refit on a common channel basis.

The prior transfer attempt (results/target_transfer.json) refused to run
because it required the ten within-animal sessions' channel identifier lists
to be LITERALLY equal, which they are not: each session's own per-day unit
yield varies (65-75 channels out of an 86-channel union), so the equality
check fails even though the sessions plainly share most of an electrode
array. This script measures the sessions' actual channel identifier
intersection directly, and if it is large enough, refits the latent model
and reruns the leave-one-session-out transfer test on that common basis
instead of demanding full equality.

Scope, stated once here and repeated in every reported number: this can only
establish transfer ACROSS SESSIONS WITHIN ONE ANIMAL. The other animal in
this dataset is recorded on a different, incompatible array and shares no
channel identifier with this one; nothing here speaks to whether a target
ranking transfers across subjects.

Three questions, in order:
  1. Is the common-basis latent fit a neutral restriction of the full-channel
     fit, or does cutting channels change each session's own within-session
     result? (checked against a fresh reproduction of the full-channel
     numbers already on disk, not against the on-disk numbers directly)
  2. Leave-one-session-out: does a target chosen on the other sessions score
     above chance on a held-out session, pooled by the same cluster bootstrap
     over sessions the delivered transfer script already uses?
  3. Is that transfer result separable from the fact that the common-basis
     channels were not picked at random -- they are exactly the channels
     that yielded units on every single day?

Reuses (no duplicated causal/geometry/statistics logic):
  - scripts.run_macaque_pfc_microstimulation_pipeline: load_macaque_pfc_microstimulation_session, crop_trial, SESSIONS,
    N_PC, DMD_RANK, N_BINS, BIN_S (the same constants the delivered pipeline
    fits with).
  - scripts.run_target_transfer: _cluster_bootstrap_over_sessions (the exact
    session-level cluster bootstrap the delivered transfer attempt built).
  - scripts.run_macaque_pfc_microstimulation_headline_robustness: _build_all_rows, _build_X,
    _col, _slope_formula, ARMS (the exact pooled-cross-fit-plus-session-
    dummies construction that produced the on-disk per-session slopes this
    script's reproduction gate checks against).
  - src.geometry.pca_decompose, src.dynamics.dmd_reconstruction_error,
    src.control.dominant_eigenmode (the same fit primitives
    build_session_features already uses -- no new geometry here).
  - src.causal.benchmark_modifiers, crossfit_nuisances, aipw_pseudo_outcome
    (the same doubly-robust machinery every other causal-targeting arm in
    this project uses).
  - src.statistics.stable_seed, paired_sign_flip_test, permutation_pvalue,
    permutation_test_twosample, minimum_detectable_paired_difference.
  - src.provenance._json_safe, checkpoint_safe, restore_checkpoint.

Does NOT modify results/target_transfer.json (that file is the record of
what the literal-equality check returned) and does NOT modify
scripts/run_target_transfer.py or scripts/run_macaque_pfc_microstimulation_pipeline.py.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_common_basis_target_transfer.py
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from geometry import pca_decompose  # noqa: E402
from dynamics import dmd_reconstruction_error  # noqa: E402
from control import dominant_eigenmode  # noqa: E402
from causal import benchmark_modifiers, crossfit_nuisances, aipw_pseudo_outcome  # noqa: E402
from statistics import (  # noqa: E402
    stable_seed, paired_sign_flip_test, permutation_test_twosample,
    minimum_detectable_paired_difference,
)
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402

import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402
import run_macaque_pfc_microstimulation_headline_robustness as headline  # noqa: E402
from run_target_transfer import _cluster_bootstrap_over_sessions  # noqa: E402

RESULTS = ROOT / "results"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_common_basis_target_transfer"
WA_SESSIONS = [s for s in macaque_pfc_microstimulation.SESSIONS if s.startswith("Wa")]

MIN_INTERSECTION_CHANNELS = 30
MIN_SESSIONS = 8
MIN_CHANNELS_PER_SESSION_FIT = max(macaque_pfc_microstimulation.N_PC + 2, 10)
MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP = 3
N_YIELD_MATCHED_DRAWS = 25
YIELD_FLOOR_FOR_CANDIDATE_POOL = 8  # same session-coverage floor as MIN_SESSIONS
N_BOOT = 2000
N_PERM = 2000
ALPHA = 0.05

SCHEMA_TAG = "v1_common_basis_target_transfer"


# =====================================================================================
# Checkpointing: temp file + os.replace, completion flag written only after a value
# returns, schema-tagged keys so a stale checkpoint from an earlier version of this
# script is a miss rather than a silent hit.
# =====================================================================================

def _checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.json"


def _load_checkpoint(name: str) -> dict:
    path = _checkpoint_path(name)
    try:
        raw = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, entry in raw.items():
        if (isinstance(entry, dict) and entry.get("complete") is True
                and entry.get("schema") == SCHEMA_TAG):
            out[key] = restore_checkpoint(entry["value"])
    return out


def _save_checkpoint_entry(name: str, cache: dict, key: str, value) -> None:
    cache[key] = value
    path = _checkpoint_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    on_disk = {k: {"complete": True, "schema": SCHEMA_TAG, "value": checkpoint_safe(v)}
               for k, v in cache.items()}
    scratch = path.with_suffix(".partial")
    scratch.write_text(json.dumps(on_disk, allow_nan=False, default=float))
    os.replace(scratch, path)


# =====================================================================================
# Channel basis
# =====================================================================================

def _session_channel_sets() -> tuple[dict[str, set[int]], list[str]]:
    """Per-session channel_ids (post shorted-channel filtering, exactly what
    load_macaque_pfc_microstimulation_session already applies -- no new filtering here), and the
    list of sessions that failed to load at all."""
    chan_sets, refused = {}, []
    for s in WA_SESSIONS:
        d = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(s, correct=True)
        if d is None:
            refused.append(s)
            continue
        chan_sets[s] = set(int(c) for c in d["channel_ids"].tolist())
    return chan_sets, refused


def _common_channel_basis(chan_sets: dict[str, set[int]]) -> set[int]:
    """The common basis is exactly the intersection of the supplied
    per-session channel identifier sets -- pulled out as its own pure
    function so this specific construction is directly testable without a
    data-mount dependency."""
    return set.intersection(*chan_sets.values()) if chan_sets else set()


def _epochs_for(cond_source: dict, cond: int, label_correct: int) -> list[tuple[np.ndarray, int, int]]:
    out = []
    for tr in cond_source["trials"]:
        if tr["stim_cond"] != cond:
            continue
        cropped = macaque_pfc_microstimulation.crop_trial(tr["spikerate"])
        if cropped is not None:
            out.append((cropped, label_correct, tr["angle_idx"]))
    return out


def _fit_common_basis_session(prefix: str, canonical_ids: list[int]) -> dict | None:
    """PCA + DMD + dominant eigenmode restricted to whichever of
    `canonical_ids` this session actually recorded that day (for the true
    intersection basis, that is always all of them; for a yield-matched
    random draw it can be fewer). V and v_star are returned embedded in the
    FULL canonical-length channel space, with an exact-zero row for any
    canonical channel this session did not record -- a session that never
    measured a channel contributes nothing to it, never a fabricated value,
    so cross-session channel-space vectors are directly comparable in one
    fixed coordinate order regardless of which sessions recorded which
    channels that day.
    """
    corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    channel_ids = corr["channel_ids"]
    pos_in_session = {int(cid): i for i, cid in enumerate(channel_ids)}
    present_canonical_idx = [j for j, cid in enumerate(canonical_ids) if cid in pos_in_session]
    if len(present_canonical_idx) < MIN_CHANNELS_PER_SESSION_FIT:
        return None
    session_cols = [pos_in_session[canonical_ids[j]] for j in present_canonical_idx]

    control_idx = corr["control_idx"]
    ctrl_epochs = []
    for tr in corr["trials"]:
        if tr["stim_cond"] != control_idx:
            continue
        cropped = macaque_pfc_microstimulation.crop_trial(tr["spikerate"][:, session_cols])
        if cropped is not None:
            ctrl_epochs.append(cropped)
    if len(ctrl_epochs) < 10:
        return None

    Z_ctrl = np.stack(ctrl_epochs, axis=0)
    C_present = len(session_cols)
    X_flat = Z_ctrl.reshape(-1, C_present)
    channel_mean = X_flat.mean(0)
    _, V_present, _ = pca_decompose(X_flat, macaque_pfc_microstimulation.N_PC)
    k = V_present.shape[1]
    Z_ctrl_mean = ((X_flat - channel_mean) @ V_present).reshape(
        Z_ctrl.shape[0], macaque_pfc_microstimulation.N_BINS, k).mean(0)
    r_use = min(macaque_pfc_microstimulation.DMD_RANK, k, macaque_pfc_microstimulation.N_BINS - 2)
    dmd = dmd_reconstruction_error(Z_ctrl_mean, r=r_use, dt=macaque_pfc_microstimulation.BIN_S)
    v_star = dominant_eigenmode(dmd["A"]).v_star

    V_canonical = np.zeros((len(canonical_ids), k))
    V_canonical[present_canonical_idx, :] = V_present

    return {
        "V": V_canonical, "v_star": v_star, "n_present": C_present,
        "present_canonical_ids": [canonical_ids[j] for j in present_canonical_idx],
    }


def _alignment_by_condition(fit: dict, corr: dict, canonical_ids: list[int]) -> dict[int, float]:
    """Per stim-condition |cos(B_hat, v_star)| in this session's own latent
    space, restricted to conditions whose stim electrode(s) are ALL present
    in this session's realisation of the canonical basis (a partial B built
    from only some of a bipolar/multi-electrode condition misrepresents the
    delivered input -- same requirement build_session_features already
    enforces for the shorted-channel exclusion)."""
    V, v_star = fit["V"], fit["v_star"]
    present_set = set(fit["present_canonical_ids"])
    control_idx = corr["control_idx"]
    align = {}
    for c, chan_ids in enumerate(corr["stim_channels"]):
        if c == control_idx or not chan_ids or not all(cid in present_set for cid in chan_ids):
            continue
        idx = [j for j, cid in enumerate(canonical_ids) if cid in chan_ids]
        B_chan = np.zeros((len(canonical_ids), 1))
        B_chan[idx, 0] = 1.0 / len(idx)
        B_lat = V.T @ B_chan
        norm = np.linalg.norm(B_lat)
        if norm < 1e-12:
            continue
        B_hat_unit = B_lat[:, 0] / norm
        align[c] = float(np.abs(B_hat_unit @ v_star))
    return align


def _rows_from_modifier(prefix: str, modifier_by_cond: dict[int, float]) -> list[dict]:
    """Trial-level rows tagged with a per-condition modifier value -- the
    same row shape build_session_features's own trial-level table construction
    builds (y, t, modifier, propensity, angle_idx), reused unchanged here for whichever
    modifier (own-session alignment_to_vstar, or a cross-session transfer
    alignment) is being scored."""
    corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(prefix, correct=True)
    err = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(prefix, correct=False)
    control_idx = corr["control_idx"]
    ctrl_all = _epochs_for(corr, control_idx, 1) + (
        _epochs_for(err, control_idx, 0) if err is not None else [])
    rows = []
    for c, mval in modifier_by_cond.items():
        stim_all = _epochs_for(corr, c, 1) + (_epochs_for(err, c, 0) if err is not None else [])
        if len(stim_all) < 5 or len(ctrl_all) < 5:
            continue
        n_stim, n_ctrl = len(stim_all), len(ctrl_all)
        propensity = n_stim / (n_stim + n_ctrl)
        for _, y, angle_idx in stim_all:
            rows.append({"y": y, "t": 1, "modifier": mval, "propensity": propensity, "angle_idx": angle_idx})
        for _, y, angle_idx in ctrl_all:
            rows.append({"y": y, "t": 0, "modifier": mval, "propensity": propensity, "angle_idx": angle_idx})
    return rows


def _slope_formula(m: np.ndarray, phi: np.ndarray) -> float:
    """Closed-form OLS slope of phi on m -- the same three-line formula
    already duplicated (deliberately, per its own docstrings) in
    src/causal._dr_slope, run_target_transfer._slope_formula, and
    run_macaque_pfc_microstimulation_headline_robustness._slope_formula, so a fourth local copy
    matches established project convention rather than importing a private
    single-purpose helper across an extra module boundary."""
    mc = m - m.mean()
    denom = (mc ** 2).sum()
    if denom < 1e-15:
        return 0.0
    return float((mc * (phi - phi.mean())).sum() / denom)


def _modifier_subset_is_not_fitted(m: np.ndarray) -> bool:
    """True exactly when _slope_formula's own degeneracy branch would fire on
    this subset (empty, or centered sum-of-squares under the same 1e-15
    floor) -- a session with zero or one surviving condition gives every one
    of its own rows an identical modifier value, so the within-session slope
    returned is not a measurement of anything, it is the fixed fallback
    value _slope_formula returns when there is no within-session contrast to
    fit a slope to. Detecting the CAUSE (near-zero variance) rather than the
    OUTCOME (slope == 0.0) is used deliberately: a genuinely fitted slope
    could in principle also land on exactly 0.0 by chance, but it could not
    also have zero within-session modifier variance.
    """
    if len(m) == 0:
        return True
    mc = m - m.mean()
    return bool((mc ** 2).sum() < 1e-15)


def _point_slope_only(y: np.ndarray, t: np.ndarray, X: np.ndarray, modifier: np.ndarray,
                       propensity: np.ndarray, rng: np.random.Generator) -> float:
    """The observed z-scored DR-interaction slope only, skipping the
    permutation null and bootstrap CI _dr_slope also computes -- used only
    for the >=25-draw null-distribution control, where only the point value
    is needed per draw and the repeated 2000-iteration inner loops would be
    pure overhead."""
    nu = crossfit_nuisances(y, t, X, propensity=propensity, rng=rng)
    phi = aipw_pseudo_outcome(y, t, nu["e_hat"], nu["mu0_hat"], nu["mu1_hat"])
    finite = np.isfinite(phi) & np.isfinite(modifier)
    phi, m = phi[finite], modifier[finite]
    sd = float(np.nanstd(m))
    if sd < 1e-12:
        return 0.0
    mz = (m - m.mean()) / sd
    return _slope_formula(mz, phi)


# =====================================================================================
# Transfer test (leave-one-session-out), one call per basis (true intersection or a
# yield-matched random draw)
# =====================================================================================

def _leave_one_out_transfer(fits: dict[str, dict], canonical_ids: list[int],
                             rng_tag: str, full_inference: bool) -> tuple[dict, dict]:
    """Leave-one-session-out: for each held-out session, the transfer target
    is the unit-norm mean of v*_chan (V @ v_star, in the shared canonical
    channel-space coordinates) over the OTHER sessions, projected into the
    held-out session's own latent space and scored there -- consistent with
    how alignment_to_vstar is scored everywhere else in this project (always
    in the stimulating session's own latent).

    full_inference=True runs benchmark_modifiers (permutation p, bootstrap
    CI) per held-out session, for the reported transfer-test result.
    full_inference=False computes only the point slope (see
    _point_slope_only), for a null-distribution draw where only the value
    is needed.

    Returns (per_session, skip_reasons) -- skip_reasons names, for every
    held-out session that did NOT get a scored slope, why not (this is the
    dominant source of session loss once the basis is restricted: a
    condition's stim electrode(s) surviving the shorted-channel exclusion is
    a different requirement from surviving the cross-session common-basis
    restriction, and a session can be left with only one, or zero,
    conditions whose stim electrode(s) fall inside the intersection -- one
    surviving condition alone gives the row-level modifier zero variance,
    which benchmark_modifiers correctly refuses to score).
    """
    v_star_chan = {s: fits[s]["V"] @ fits[s]["v_star"] for s in fits}
    per_session, skip_reasons = {}, {}
    for held_out in fits:
        others = [s for s in fits if s != held_out]
        if len(others) < 2:
            skip_reasons[held_out] = "fewer than 2 other sessions available to build a transfer target"
            continue
        mean_chan = np.mean([v_star_chan[s] for s in others], axis=0)
        norm = np.linalg.norm(mean_chan)
        if norm < 1e-12:
            skip_reasons[held_out] = "cross-session mean channel-space direction is exactly zero"
            continue
        transfer_target_chan = mean_chan / norm
        V_ho = fits[held_out]["V"]
        transfer_target_latent = V_ho.T @ transfer_target_chan
        tnorm = np.linalg.norm(transfer_target_latent)
        if tnorm < 1e-12:
            skip_reasons[held_out] = "transfer target projects to exactly zero in this session's own latent space"
            continue
        transfer_target_latent /= tnorm

        corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(held_out, correct=True)
        present_set = set(fits[held_out]["present_canonical_ids"])
        control_idx = corr["control_idx"]
        n_cond_total = sum(1 for c in range(len(corr["stim_channels"])) if c != control_idx)
        talign_by_cond = {}
        for c, chan_ids in enumerate(corr["stim_channels"]):
            if c == control_idx or not chan_ids or not all(cid in present_set for cid in chan_ids):
                continue
            idx = [j for j, cid in enumerate(canonical_ids) if cid in chan_ids]
            B_chan = np.zeros((len(canonical_ids), 1))
            B_chan[idx, 0] = 1.0 / len(idx)
            B_lat = V_ho.T @ B_chan
            bnorm = np.linalg.norm(B_lat)
            if bnorm < 1e-12:
                continue
            B_hat_unit = B_lat[:, 0] / bnorm
            talign_by_cond[c] = float(np.abs(B_hat_unit @ transfer_target_latent))

        if len(talign_by_cond) < 2:
            skip_reasons[held_out] = (
                f"only {len(talign_by_cond)} of {n_cond_total} stim conditions have every stim "
                "electrode inside the common basis; fewer than 2 surviving conditions gives the "
                "row-level modifier zero variance"
            )
            continue

        rows = _rows_from_modifier(held_out, talign_by_cond)
        if len(rows) < 20:
            skip_reasons[held_out] = f"only {len(rows)} rows after the surviving-condition restriction (need >=20)"
            continue
        y = np.array([r["y"] for r in rows], dtype=float)
        t = np.array([r["t"] for r in rows], dtype=int)
        modifier = np.array([r["modifier"] for r in rows], dtype=float)
        propensity = np.array([r["propensity"] for r in rows], dtype=float)
        angle_idx = np.array([r["angle_idx"] for r in rows], dtype=int)
        X = np.eye(angle_idx.max() + 1)[angle_idx]

        sess_rng = np.random.default_rng(stable_seed(f"{rng_tag}_{held_out}"))
        if full_inference:
            bench = benchmark_modifiers(y, t, X, modifiers={"transfer_alignment": modifier},
                                        propensity=propensity, n_perm=N_PERM, rng=sess_rng)
            if "transfer_alignment" in bench.get("excluded", {}):
                skip_reasons[held_out] = f"benchmark_modifiers excluded the modifier: {bench['excluded']['transfer_alignment']['reason']}"
                continue
            row = bench["leaderboard"]["transfer_alignment"]
            per_session[held_out] = {"transfer_slope": row["slope"], "slope_ci_lo": row["slope_ci_lo"],
                                      "slope_ci_hi": row["slope_ci_hi"], "p_value": row["p_value"],
                                      "n": row["n"], "n_conditions_surviving_basis": len(talign_by_cond),
                                      "n_conditions_total": n_cond_total}
        else:
            slope = _point_slope_only(y, t, X, modifier, propensity, sess_rng)
            per_session[held_out] = {"transfer_slope": slope}

    return per_session, skip_reasons


# =====================================================================================
# Precondition
# =====================================================================================

def _run_precondition() -> dict:
    chan_sets, refused = _session_channel_sets()
    per_session_counts = {s: len(v) for s, v in chan_sets.items()}
    n_seen = len(WA_SESSIONS)
    n_loaded = len(chan_sets)
    intersection = _common_channel_basis(chan_sets)
    union = set.union(*chan_sets.values()) if chan_sets else set()
    zero_drop = {
        "n_sessions_seen": n_seen, "n_sessions_loaded": n_loaded,
        "n_sessions_refused": len(refused),
        "refusals_by_reason": {"load_macaque_pfc_microstimulation_session_returned_none": len(refused)} if refused else {},
        "reconciles": bool(n_seen == n_loaded + len(refused)),
    }
    passes = len(intersection) >= MIN_INTERSECTION_CHANNELS and n_loaded >= MIN_SESSIONS
    return {
        "per_session_channel_counts": per_session_counts,
        "intersection_channel_ids": sorted(intersection),
        "intersection_size": len(intersection),
        "union_channel_ids": sorted(union),
        "union_size": len(union),
        "zero_drop_accounting": zero_drop,
        "min_intersection_channels_required": MIN_INTERSECTION_CHANNELS,
        "min_sessions_required": MIN_SESSIONS,
        "passes": passes,
    }


# =====================================================================================
# Reproduction gate: recompute the delivered full-channel per-session vstar_alignment
# slopes fresh (same code, same seed) before trusting either them or the common-basis
# comparison against them.
# =====================================================================================

def _reproduce_full_channel_slopes() -> dict:
    all_rows, session_order = headline._build_all_rows()
    n_sessions = len(session_order)
    y = headline._col(all_rows, "y")
    t = headline._col(all_rows, "t").astype(int)
    vstar_mod = headline._col(all_rows, "modifier")
    min_energy_mod = headline._col(all_rows, "min_energy_dir_alignment")
    session_mean_mod = headline._col(all_rows, "session_mean_vstar_scalar")
    align_s_m2_mod = headline._col(all_rows, "align_s_m2")
    align_s_m3_mod = headline._col(all_rows, "align_s_m3")
    propensity = headline._col(all_rows, "propensity")
    angle_idx = headline._col(all_rows, "angle_idx").astype(int)
    session_idx = headline._col(all_rows, "session_idx").astype(int)
    X = headline._build_X(all_rows, angle_idx, session_idx, n_sessions)

    bench_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_headline_robustness"))
    bench = benchmark_modifiers(
        y, t, X, modifiers={"vstar_alignment": vstar_mod, "min_energy_dir_alignment": min_energy_mod,
                            "session_mean_vstar_scalar": session_mean_mod,
                            "align_s_m2": align_s_m2_mod, "align_s_m3": align_s_m3_mod},
        propensity=propensity, n_perm=2000, rng=bench_rng,
    )
    phi = bench["phi"]
    z_mod = bench["z_modifiers"]["vstar_alignment"]

    delivered_path = RESULTS / "macaque_pfc_microstimulation_headline_robustness.json"
    delivered = json.loads(delivered_path.read_text())
    delivered_per_session = delivered["per_session"]

    per_session = {}
    n_reproduced = n_mismatched = n_skipped = 0
    for s in range(n_sessions):
        mask = session_idx == s
        fresh = _slope_formula(z_mod[mask], phi[mask])
        name = session_order[s]
        disk_entry = delivered_per_session[name]["vstar_alignment"]
        disk = disk_entry["slope"]
        # "fitted" is the FRESH side's own determination (degenerate-modifier check on
        # this run's z_mod subset), independent of what the delivered record says --
        # unaffected by whether the delivered session was excluded.
        fitted = not _modifier_subset_is_not_fitted(z_mod[mask])
        if disk_entry.get("status") != "fitted" or disk is None:
            # The delivered producer already determined this session has no
            # within-session regressor variance and recorded slope=None with a
            # reason -- there is nothing to difference it against, so this
            # session is SKIPPED rather than compared, and the skip is recorded
            # with its reason (never silently dropped).
            per_session[name] = {
                "fresh_slope": fresh, "delivered_slope": None, "diff": None,
                "reproduction_status": "skipped_source_has_no_fit",
                "skip_reason": disk_entry.get("reason", "delivered session carries no fitted slope"),
                "fitted": fitted,
            }
            n_skipped += 1
            continue
        diff = fresh - disk
        reproduced = bool(abs(diff) < 1e-6)
        per_session[name] = {
            "fresh_slope": fresh, "delivered_slope": disk, "diff": diff,
            "reproduced_exactly": reproduced,
            "reproduction_status": "reproduced_exactly" if reproduced else "mismatched",
            "fitted": fitted,
        }
        if reproduced:
            n_reproduced += 1
        else:
            n_mismatched += 1
    all_exact = n_mismatched == 0
    return {
        "per_session": per_session, "reproduced_exactly_among_fitted_sources": all_exact,
        "reproduction_counts": {"reproduced_exactly": n_reproduced, "mismatched": n_mismatched,
                                "skipped_source_has_no_fit": n_skipped},
        "n_sessions": n_sessions, "session_order": session_order,
        "phi": phi, "z_mod": z_mod, "session_idx": session_idx,
    }


# =====================================================================================
# Neutral-restriction check: fresh full-channel slopes vs common-basis slopes, same
# pooled-cross-fit-plus-session-dummies construction as the reproduction above.
# =====================================================================================

def _common_basis_pooled_slopes(canonical_ids: list[int]) -> dict:
    fits = {}
    for s in WA_SESSIONS:
        f = _fit_common_basis_session(s, canonical_ids)
        if f is not None:
            fits[s] = f
    session_order = sorted(fits.keys())
    all_rows = []
    for si, s in enumerate(session_order):
        corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(s, correct=True)
        align = _alignment_by_condition(fits[s], corr, canonical_ids)
        rows = _rows_from_modifier(s, align)
        for r in rows:
            r["session_idx"] = si
        all_rows.extend(rows)

    n_sessions = len(session_order)
    y = np.array([r["y"] for r in all_rows], dtype=float)
    t = np.array([r["t"] for r in all_rows], dtype=int)
    modifier = np.array([r["modifier"] for r in all_rows], dtype=float)
    propensity = np.array([r["propensity"] for r in all_rows], dtype=float)
    angle_idx = np.array([r["angle_idx"] for r in all_rows], dtype=int)
    session_idx = np.array([r["session_idx"] for r in all_rows], dtype=int)
    X = headline._build_X(all_rows, angle_idx, session_idx, n_sessions)

    bench_rng = np.random.default_rng(stable_seed("common_basis_target_transfer_pooled"))
    bench = benchmark_modifiers(y, t, X, modifiers={"alignment_to_vstar_common_basis": modifier},
                                propensity=propensity, n_perm=N_PERM, rng=bench_rng)
    phi = bench["phi"]
    z_mod = bench["z_modifiers"]["alignment_to_vstar_common_basis"]

    per_session_slope, per_session_fitted = {}, {}
    for si, s in enumerate(session_order):
        mask = session_idx == si
        per_session_slope[s] = _slope_formula(z_mod[mask], phi[mask])
        per_session_fitted[s] = not _modifier_subset_is_not_fitted(z_mod[mask])

    return {"fits": fits, "session_order": session_order, "per_session_slope": per_session_slope,
            "per_session_fitted": per_session_fitted}


# =====================================================================================
# Selection controls
# =====================================================================================

def _yield_and_firing_rate_control(intersection_ids: set[int], union_ids: set[int]) -> dict:
    excluded_ids = union_ids - intersection_ids
    chan_sets, _ = _session_channel_sets()
    session_order = sorted(chan_sets.keys())

    yield_count: dict[int, int] = {}
    rate_values: dict[int, list[float]] = {}
    for s in session_order:
        corr = macaque_pfc_microstimulation.load_macaque_pfc_microstimulation_session(s, correct=True)
        control_idx = corr["control_idx"]
        channel_ids = corr["channel_ids"]
        pos = {int(cid): i for i, cid in enumerate(channel_ids)}
        ctrl_mats = [tr["spikerate"] for tr in corr["trials"] if tr["stim_cond"] == control_idx]
        if not ctrl_mats:
            continue
        mean_rate_this_session = np.mean(np.concatenate(ctrl_mats, axis=0), axis=0)  # (C,)
        for cid, i in pos.items():
            yield_count[cid] = yield_count.get(cid, 0) + 1
            rate_values.setdefault(cid, []).append(float(mean_rate_this_session[i]))

    inter_yield = np.array([yield_count[c] for c in intersection_ids if c in yield_count])
    excl_yield = np.array([yield_count[c] for c in excluded_ids if c in yield_count])
    inter_rate = np.array([np.mean(rate_values[c]) for c in intersection_ids if c in rate_values])
    excl_rate = np.array([np.mean(rate_values[c]) for c in excluded_ids if c in rate_values])

    rng = np.random.default_rng(stable_seed("common_basis_target_transfer_selection_control"))
    yield_obs, yield_p = permutation_test_twosample(inter_yield, excl_yield, n_perm=N_PERM, rng=rng)
    rate_obs, rate_p = permutation_test_twosample(inter_rate, excl_rate, n_perm=N_PERM, rng=rng)

    return {
        "note": ("channel-session presence count is compared for completeness, but every "
                 "intersection channel is present in every session BY DEFINITION and every "
                 "excluded channel is absent from at least one, so that comparison is "
                 "tautological, not an independent finding; firing rate is the informative "
                 "comparison here"),
        "n_intersection_channels": int(len(inter_rate)), "n_excluded_channels": int(len(excl_rate)),
        "session_presence_count": {"intersection_mean": float(inter_yield.mean()),
                                   "excluded_mean": float(excl_yield.mean()),
                                   "diff": yield_obs, "p_value": yield_p},
        "control_epoch_firing_rate": {"intersection_mean": float(inter_rate.mean()),
                                      "excluded_mean": float(excl_rate.mean()),
                                      "diff": rate_obs, "p_value": rate_p},
    }


def _yield_matched_candidate_pool() -> tuple[list[int], dict[str, set[int]]]:
    chan_sets, _ = _session_channel_sets()
    counts: dict[int, int] = {}
    for ids in chan_sets.values():
        for c in ids:
            counts[c] = counts.get(c, 0) + 1
    pool = sorted(c for c, n in counts.items() if n >= YIELD_FLOOR_FOR_CANDIDATE_POOL)
    return pool, chan_sets


def _yield_matched_null_draws(n_channels: int, true_transfer_mean: float,
                              checkpoint_cache: dict) -> dict:
    """>=25 random draws of `n_channels` channels from the pool of channels
    that individually clear the same session-coverage floor
    (YIELD_FLOOR_FOR_CANDIDATE_POOL) the true intersection basis was held to,
    each seeded by a stable hash of a descriptive tag. Every session keeps
    whichever of the drawn channels it actually recorded that day (see
    _fit_common_basis_session); no session is fabricated data or dropped
    just because a handful of the drawn channels are missing on that day.
    """
    pool, _ = _yield_matched_candidate_pool()
    draw_means = []
    for i in range(N_YIELD_MATCHED_DRAWS):
        key = f"draw_{i}_n{n_channels}"
        if key in checkpoint_cache:
            draw_means.append(checkpoint_cache[key]["mean"])
            continue
        seed_tag = f"common_basis_target_transfer_yield_matched_draw_{i}_n{n_channels}"
        rng = np.random.default_rng(stable_seed(seed_tag))
        draw_ids = sorted(int(c) for c in rng.choice(pool, size=n_channels, replace=False))
        fits = {}
        for s in WA_SESSIONS:
            f = _fit_common_basis_session(s, draw_ids)
            if f is not None:
                fits[s] = f
        if len(fits) < MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP:
            entry = {"mean": float("nan"), "n_sessions": len(fits), "draw_ids": draw_ids}
            _save_checkpoint_entry("yield_matched_draws", checkpoint_cache, key, entry)
            continue
        per_session, _skip = _leave_one_out_transfer(fits, draw_ids, seed_tag, full_inference=False)
        slopes = np.array([v["transfer_slope"] for v in per_session.values()])
        mean_val = float(slopes.mean()) if len(slopes) >= MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP else float("nan")
        entry = {"mean": mean_val, "n_sessions": len(per_session), "draw_ids": draw_ids}
        _save_checkpoint_entry("yield_matched_draws", checkpoint_cache, key, entry)
        draw_means.append(mean_val)

    null_dist = np.array([m for m in draw_means if np.isfinite(m)])
    if len(null_dist) < 5:
        return {"status": "not_computable", "n_usable_draws": int(len(null_dist)),
                "reason": "fewer than 5 of the requested draws produced a usable leave-one-out slope"}
    p = 2.0 * min(float((null_dist <= true_transfer_mean).mean()),
                  float((null_dist >= true_transfer_mean).mean()))
    p = min(p, 1.0)
    return {
        "status": "computed", "n_draws_requested": N_YIELD_MATCHED_DRAWS,
        "n_usable_draws": int(len(null_dist)),
        "candidate_pool_size": len(pool), "candidate_pool_yield_floor": YIELD_FLOOR_FOR_CANDIDATE_POOL,
        "null_distribution_mean": float(null_dist.mean()), "null_distribution_sd": float(null_dist.std(ddof=1)),
        "null_distribution": null_dist.tolist(),
        "true_intersection_transfer_mean": true_transfer_mean,
        "p_value_true_vs_null": p,
        "not_separable_from_yield_selection": bool(p >= ALPHA),
    }


# =====================================================================================
# Driver
# =====================================================================================

def main() -> None:
    t0 = time.time()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict = {
        "scope_limit": (
            "this establishes cross-session stimulation-target transfer WITHIN ONE "
            "animal only; the other animal in this dataset is a different, incompatible "
            "array and shares no channel identifier with this one, so nothing here speaks "
            "to whether a target ranking transfers across subjects"
        ),
        "sessions_considered": WA_SESSIONS,
    }
    branches_fired = []

    print("Measuring the actual channel identifier intersection across the within-animal sessions ...")
    precondition = _run_precondition()
    out["channel_basis"] = precondition
    print(f"  per-session channel counts: {precondition['per_session_channel_counts']}")
    print(f"  intersection size = {precondition['intersection_size']}, union size = {precondition['union_size']}")

    if not precondition["passes"]:
        branches_fired.append("common_channel_basis_too_small_to_support_a_transfer_test")
        out["branches_fired"] = branches_fired
        out["status"] = "complete"
        out["wall_clock_s"] = time.time() - t0
        with open(RESULTS / "common_basis_target_transfer.json", "w") as f:
            json.dump(_json_safe(out), f, indent=2, allow_nan=False)
        print("Common basis too small -- stopping (this branch alone is reported).")
        return

    canonical_ids = precondition["intersection_channel_ids"]

    print("\nReproduction gate: recomputing the delivered full-channel per-session "
          "vstar_alignment slopes fresh, same code and seed as the delivered pipeline ...")
    repro = _reproduce_full_channel_slopes()
    out["reproduction_gate"] = {
        "per_session": {s: {k: v for k, v in vals.items()} for s, vals in repro["per_session"].items()},
        "reproduced_exactly_among_fitted_sources": repro["reproduced_exactly_among_fitted_sources"],
        "reproduction_counts": repro["reproduction_counts"],
    }
    rc = repro["reproduction_counts"]
    print(f"  reproduced_exactly_among_fitted_sources = {repro['reproduced_exactly_among_fitted_sources']} "
          f"(reproduced {rc['reproduced_exactly']}, mismatched {rc['mismatched']}, "
          f"skipped for want of a fit {rc['skipped_source_has_no_fit']})")

    print("\nRefitting the latent model and dominant eigenmode on the common basis alone ...")
    common = _common_basis_pooled_slopes(canonical_ids)
    sessions_common = sorted(s for s in WA_SESSIONS if s in common["per_session_slope"] and s in repro["per_session"])

    # A session's own-basis or full-channel slope is only a measurement if that side had
    # at least two surviving conditions to fit a within-session contrast to (see
    # _modifier_subset_is_not_fitted); an exact 0.0 with no surviving contrast is the
    # fallback value the closed-form slope returns, never a fitted zero, on EITHER side.
    not_fitted_common_basis = [s for s in sessions_common if not common["per_session_fitted"][s]]
    not_fitted_full_channel = [s for s in sessions_common if not repro["per_session"][s]["fitted"]]
    both_fitted = [s for s in sessions_common
                  if common["per_session_fitted"][s] and repro["per_session"][s]["fitted"]]

    per_session_common_basis_slope = {s: common["per_session_slope"][s] for s in sessions_common}
    per_session_full_channel_slope_fresh = {s: repro["per_session"][s]["fresh_slope"] for s in sessions_common}
    per_session_fitted_common_basis = {s: common["per_session_fitted"][s] for s in sessions_common}
    per_session_fitted_full_channel = {s: repro["per_session"][s]["fitted"] for s in sessions_common}

    MIN_SESSIONS_WITH_A_GENUINE_FIT_ON_BOTH_SIDES = 3
    if len(both_fitted) >= MIN_SESSIONS_WITH_A_GENUINE_FIT_ON_BOTH_SIDES:
        a_common = np.array([common["per_session_slope"][s] for s in both_fitted])
        a_full = np.array([repro["per_session"][s]["fresh_slope"] for s in both_fitted])
        paired = paired_sign_flip_test(a_common, a_full, alternative="two-sided",
                                       rng=np.random.default_rng(stable_seed("common_basis_target_transfer_paired")))
        restriction_changes_result = bool(paired["p_value"] < ALPHA)
        neutral_restriction_status = "evaluated"
        paired_test_out = {"mean_diff": paired["mean_diff"], "p_value": paired["p_value"],
                           "ci_lower": paired["ci_lower"], "ci_upper": paired["ci_upper"],
                           "n_sessions": paired["n"]}
        print(f"  paired mean diff = {paired['mean_diff']:+.4f}, p = {paired['p_value']:.4f} "
              f"(n={paired['n']} sessions with a genuine fit on both sides)")
    else:
        # No determination is possible here, so this stays null rather than False: False would assert
        # that the restriction leaves the within-session result unchanged, which is the very claim the
        # not-evaluable status withholds.
        restriction_changes_result = None
        neutral_restriction_status = "not_evaluable_too_few_sessions_with_a_genuine_fit_on_both_sides"
        paired_test_out = {"status": "not_computed", "n_sessions_with_a_genuine_fit_on_both_sides": len(both_fitted)}
        print(f"  neutral-restriction check not evaluable: only {len(both_fitted)} of "
              f"{len(sessions_common)} sessions have a genuine fitted slope on both sides")

    out["neutral_restriction_check"] = {
        "status": neutral_restriction_status,
        "per_session_common_basis_slope": per_session_common_basis_slope,
        "per_session_full_channel_slope_fresh": per_session_full_channel_slope_fresh,
        "per_session_fitted_common_basis": per_session_fitted_common_basis,
        "per_session_fitted_full_channel": per_session_fitted_full_channel,
        "n_sessions_considered": len(sessions_common),
        "n_not_fitted_common_basis": len(not_fitted_common_basis),
        "n_not_fitted_full_channel": len(not_fitted_full_channel),
        "not_fitted_common_basis_sessions": not_fitted_common_basis,
        "not_fitted_full_channel_sessions": not_fitted_full_channel,
        "n_sessions_with_a_genuine_fit_on_both_sides": len(both_fitted),
        "sessions_with_a_genuine_fit_on_both_sides": both_fitted,
        "paired_session_level_test": paired_test_out,
        "restriction_changes_the_within_session_result": restriction_changes_result,
    }
    print(f"  not-fitted sessions: common-basis side {len(not_fitted_common_basis)}/{len(sessions_common)}, "
          f"full-channel side {len(not_fitted_full_channel)}/{len(sessions_common)}")
    if restriction_changes_result:
        branches_fired.append("restriction_to_the_common_basis_changes_the_within_session_result")

    print("\nLeave-one-session-out transfer test on the true common basis ...")
    fits = common["fits"]
    transfer_per_session, transfer_skip_reasons = _leave_one_out_transfer(
        fits, canonical_ids, "common_basis_target_transfer", full_inference=True)
    own_slopes_arr = np.array([common["per_session_slope"][s] for s in transfer_per_session])
    transfer_slopes_arr = np.array([v["transfer_slope"] for v in transfer_per_session.values()])
    n_scored = len(transfer_slopes_arr)
    print(f"  {n_scored}/{len(fits)} held-out sessions scored; skipped: {transfer_skip_reasons}")

    if n_scored < MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP:
        # The channel-count precondition passed (recorded in out["channel_basis"], kept
        # right here so the two are never read apart): the basis has enough CHANNELS and
        # enough SESSIONS. What collapses is STIMULATION-CONDITION coverage -- a condition
        # only survives the restriction if every one of its stim electrodes lands inside
        # the always-present set, and skip_reasons shows most sessions keep 0 or 1 of
        # theirs. That is a distinct failure mode from the basis being too small, so it is
        # named separately rather than reusing the channel-count branch.
        out["transfer_test"] = {"status": "infeasible",
                                "reason": f"only {n_scored} held-out sessions scored, too few for a cluster bootstrap",
                                "per_session": transfer_per_session,
                                "skip_reasons": transfer_skip_reasons,
                                "channel_count_precondition": {
                                    "intersection_size": precondition["intersection_size"],
                                    "n_sessions": precondition["zero_drop_accounting"]["n_sessions_loaded"],
                                    "passes": precondition["passes"],
                                }}
        branches_fired.append("common_basis_preserves_channels_but_not_stimulation_condition_coverage")
    else:
        boot_rng = np.random.default_rng(stable_seed("common_basis_target_transfer_cluster_bootstrap"))
        transfer_cluster = _cluster_bootstrap_over_sessions(transfer_slopes_arr, N_BOOT, boot_rng)
        own_cluster = _cluster_bootstrap_over_sessions(own_slopes_arr, N_BOOT, boot_rng)
        retention = (transfer_cluster["mean"] / own_cluster["mean"]) if abs(own_cluster["mean"]) > 1e-9 else float("nan")
        mdd = minimum_detectable_paired_difference(transfer_slopes_arr, alpha=ALPHA, power=0.80)
        reference_value = abs(own_cluster["mean"])

        print(f"  transfer: mean={transfer_cluster['mean']:+.4f} "
              f"CI[{transfer_cluster['ci_lo']:+.4f},{transfer_cluster['ci_hi']:+.4f}] "
              f"p={transfer_cluster['p_value']:.4f}  n_held_out={n_scored}")
        print(f"  own-session (common basis) reference mean = {own_cluster['mean']:+.4f}")
        print(f"  mdd@80% power = {mdd.get('mdd')}")

        transfers = transfer_cluster["ci_lo"] > 0 and transfer_cluster["p_value"] < ALPHA
        does_not_transfer_is_powered = (
            not transfers and mdd.get("status") == "computed" and mdd["mdd"] < reference_value
        )
        out["transfer_test"] = {
            "status": "complete",
            "per_held_out_session": transfer_per_session,
            "n_held_out_sessions_scored": n_scored,
            "skip_reasons": transfer_skip_reasons,
            "transfer_cluster_bootstrap": transfer_cluster,
            "own_session_common_basis_cluster_bootstrap": own_cluster,
            "retention": retention,
            "minimum_detectable_difference_at_80pct_power": mdd,
            "named_reference_for_does_not_transfer": (
                "the mean own-session common-basis vstar_alignment slope over the same "
                "held-out sessions, pooled by the same cluster bootstrap"
            ),
            "reference_value": reference_value,
            "transfers_significantly": bool(transfers),
            "does_not_transfer_is_a_powered_null": bool(does_not_transfer_is_powered),
        }
        if transfers:
            branches_fired.append("a_target_chosen_on_held_in_sessions_transfers_to_held_out_sessions")
        elif does_not_transfer_is_powered:
            branches_fired.append("a_target_chosen_on_held_in_sessions_does_not_transfer")
        else:
            branches_fired.append("inconclusive_below_detection_floor")

        print("\nSelection controls ...")
        yield_rate = _yield_and_firing_rate_control(set(canonical_ids), set(precondition["union_channel_ids"]))
        print(f"  firing-rate diff (intersection - excluded) = {yield_rate['control_epoch_firing_rate']['diff']:+.4f} "
              f"p={yield_rate['control_epoch_firing_rate']['p_value']:.4f}")

        checkpoint_cache = _load_checkpoint("yield_matched_draws")
        null_control = _yield_matched_null_draws(len(canonical_ids), transfer_cluster["mean"], checkpoint_cache)
        if null_control.get("status") == "computed":
            print(f"  yield-matched null: {null_control['n_usable_draws']} usable draws, "
                  f"mean={null_control['null_distribution_mean']:+.4f}, "
                  f"true={transfer_cluster['mean']:+.4f}, p={null_control['p_value_true_vs_null']:.4f}")
            if null_control["not_separable_from_yield_selection"]:
                branches_fired.append("common_basis_effect_not_separable_from_channel_yield_selection")
        out["selection_controls"] = {
            "channel_property_comparison": yield_rate,
            "yield_matched_random_subset_null": null_control,
        }

    out["branches_fired"] = sorted(set(branches_fired))
    out["status"] = "complete"
    out["wall_clock_s"] = time.time() - t0
    with open(RESULTS / "common_basis_target_transfer.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    print(f"\nBranches fired: {out['branches_fired']}")
    print(f"Wall clock: {out['wall_clock_s']:.1f}s")
    print("Saved results/common_basis_target_transfer.json")


if __name__ == "__main__":
    main()
