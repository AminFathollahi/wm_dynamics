#!/usr/bin/env python3
"""Is the displacement a stimulation site produces a reproducible property of
that site, or a property of the day?

For each session, in that session's own control-fit latent space (or, for
the across-session comparison, in the raw firing-rate space restricted to
the channels shared by both sessions of a repeat), this script measures each
stimulation site's displacement of the population state away from the
no-stimulation control condition at matched target angle -- a magnitude
(normalised by that session's own control-trial-to-trial spread) and a unit
direction vector.

Two reliability tiers, each with its own pre-declared decision rule (fixed
below, before any cosine was computed):

  within-session  -- split each session's trials into two halves by their
                      order of occurrence within each (site, angle) cell
                      (the only trial order the raw files preserve after
                      grouping by condition), estimate each site's direction
                      in each half, and ask whether the same-site half-vs-
                      half cosine exceeds the different-site cosine.
  across-session  -- for the three channel sets that recur across a pair of
                      sessions in the same animal, restrict to the channels
                      recorded in BOTH sessions, estimate each site's
                      direction in each session on that shared basis, and
                      ask the same question across the pair.

One amplitude per session in this corpus: this measurement can speak to
WHERE a site displaces the state, never to how the displacement scales with
stimulation intensity.

Reuses scripts/run_macaque_pfc_microstimulation_pipeline.py's session loader
and crop window (no second signal path) and src/geometry.py's PCA and
src/statistics.py's permutation/MDD primitives (no new statistical
machinery). The only new logic is the site-identity label-shuffle test
itself, which does not exist elsewhere in the project.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_macaque_pfc_microstimulation_site_reproducibility.py
"""
from __future__ import annotations

import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from geometry import pca_decompose
from statistics import stable_seed, permutation_pvalue, minimum_detectable_paired_difference
from io_utils import locked_json_update
from provenance import _json_safe, git_commit

from run_macaque_pfc_microstimulation_pipeline import (
    load_macaque_pfc_microstimulation_session, crop_trial, SESSIONS, N_PC,
)

RESULTS = ROOT / "results"

# ── Named thresholds (fixed before any cosine or displacement was computed) ──
MIN_TRIALS_PER_ARM = 3        # a (condition, angle) cell needs this many valid
                               # trials in BOTH the site and the control arm
                               # to contribute to a displacement estimate.
MIN_CONTROL_TRIALS_FOR_FIT = 10   # floor for fitting a session's own latent basis.
MIN_SHARED_CHANNELS = 3       # floor for a repeat pair's shared-channel basis
                               # to support a directional comparison at all.
MDD_REFERENCE = 0.14          # this project's standard cosine-scale reference
                               # bound for calling a permutation null "powered."
N_PERM = 10000
N_PC_LATENT = N_PC            # reuse the causal pipeline's PC count, capped
                               # per-session by pca_decompose itself.

SECOND_ANIMAL_SESSION = "Sa210311_s224"
PRIMARY_ANIMAL_SESSIONS = [s for s in SESSIONS if s != SECOND_ANIMAL_SESSION]

REPEAT_PAIRS = [
    ("Wa220805_s553", "Wa220809_s555"),
    ("Wa220808_s554", "Wa220810_s556"),
    ("Wa220811_s557", "Wa220812_s558"),
]

BRANCH_SITE_SPECIFIC = "site_identity_determines_the_displacement"
BRANCH_POWERED_NULL = "no_site_specificity_above_the_reported_bound"
BRANCH_UNDERPOWERED = "underpowered_to_ask"
BRANCH_INFEASIBLE = "not_computable_from_this_recording"


# ── Trial-level state and per-(condition, angle) grouping ─────────────────────

def _trial_state_vector(mat: np.ndarray) -> np.ndarray | None:
    """(T, C) raw spikerate -> the trial's population state: the mean firing
    rate over the common cropped analysis window, one point per trial."""
    cropped = crop_trial(mat)
    return None if cropped is None else cropped.mean(axis=0)


def _trials_by_cond_angle(session: dict) -> dict[tuple[int, int], list[np.ndarray]]:
    """dict[(stim_cond, angle_idx)] -> raw (T, C) trial arrays, in the order
    the loader appended them -- which is the file's own per-condition cell
    order, the only trial order recoverable once the corpus's own storage
    format has grouped trials by condition."""
    table: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for tr in session["trials"]:
        table[(tr["stim_cond"], tr["angle_idx"])].append(tr["spikerate"])
    return table


def _split_table_in_half(table: dict) -> tuple[dict, dict]:
    half_a: dict = {}
    half_b: dict = {}
    for key, raws in table.items():
        mid = len(raws) // 2
        half_a[key] = raws[:mid]
        half_b[key] = raws[mid:]
    return half_a, half_b


def _condition_cell_ledger(prefix: str, table: dict, control_cond: int) -> list[dict]:
    entries = []
    for (c, a), raws in table.items():
        n_valid = sum(1 for mat in raws if _trial_state_vector(mat) is not None)
        analysed = n_valid >= MIN_TRIALS_PER_ARM
        entries.append({
            "session": prefix, "cond_index": int(c), "is_control": bool(c == control_cond),
            "angle_index": int(a), "n_trials_seen": len(raws),
            "n_trials_after_window_crop": int(n_valid),
            "status": "analysed" if analysed else "refused",
            "reason": (None if analysed else
                       f"fewer than {MIN_TRIALS_PER_ARM} trials survive the common "
                       "analysis window after cropping"),
        })
    return entries


# ── Site displacement in a caller-supplied feature space ──────────────────────

def _site_displacement(table: dict, control_cond: int, site_cond: int,
                        angle_ids: list[int], project) -> dict:
    """Displacement of the site condition from the control condition at
    matched target angle, in the space `project` maps a raw C-vector into.
    Averages the per-angle delta over every angle with enough trials in both
    arms -- never pools trials across angles."""
    deltas = []
    n_angles_seen = 0
    for a in angle_ids:
        n_angles_seen += 1
        site_states = [project(v) for v in
                       (_trial_state_vector(m) for m in table.get((site_cond, a), []))
                       if v is not None]
        ctrl_states = [project(v) for v in
                       (_trial_state_vector(m) for m in table.get((control_cond, a), []))
                       if v is not None]
        if len(site_states) >= MIN_TRIALS_PER_ARM and len(ctrl_states) >= MIN_TRIALS_PER_ARM:
            deltas.append(np.mean(site_states, axis=0) - np.mean(ctrl_states, axis=0))
    if not deltas:
        return {"status": "refused",
                "reason": f"no target angle had >= {MIN_TRIALS_PER_ARM} trials in both "
                          "the site and control arm on this feature basis"}
    disp = np.mean(deltas, axis=0)
    norm = float(np.linalg.norm(disp))
    if norm < 1e-12:
        return {"status": "refused", "reason": "degenerate (zero-norm) displacement vector"}
    return {"status": "computed", "direction": disp / norm, "raw_magnitude": norm,
            "n_angles_used": len(deltas)}


def _control_residual_sd(table: dict, control_cond: int, angle_ids: list[int], project) -> float | None:
    """Scalar trial-to-trial spread of the control condition in `project`'s
    space, with each angle's own mean removed first so angle-driven structure
    is not counted as noise."""
    residuals = []
    for a in angle_ids:
        states = [project(v) for v in
                  (_trial_state_vector(m) for m in table.get((control_cond, a), []))
                  if v is not None]
        if len(states) >= 2:
            arr = np.stack(states)
            residuals.append(arr - arr.mean(axis=0))
    if not residuals:
        return None
    r = np.concatenate(residuals, axis=0)
    sd_vec = r.std(axis=0, ddof=1)
    return float(np.linalg.norm(sd_vec))


def _fit_latent_basis(session: dict, control_cond: int) -> tuple[np.ndarray, float, int] | None:
    states = [v for v in (_trial_state_vector(tr["spikerate"]) for tr in session["trials"]
                          if tr["stim_cond"] == control_cond) if v is not None]
    if len(states) < MIN_CONTROL_TRIALS_FOR_FIT:
        return None
    X = np.stack(states)
    _, V, var_ratio = pca_decompose(X, N_PC_LATENT)
    return V, float(var_ratio.sum()), len(states)


# ── The site-identity label-shuffle test (shared by both tiers) ───────────────

def _decision_branch(p_value: float, mdd: dict) -> str:
    if p_value < 0.05:
        return BRANCH_SITE_SPECIFIC
    if mdd.get("status") == "computed":
        return BRANCH_POWERED_NULL if mdd["mdd"] < MDD_REFERENCE else BRANCH_UNDERPOWERED
    return BRANCH_UNDERPOWERED


def site_identity_contrast(units: dict[str, dict[str, dict]], rng: np.random.Generator,
                            n_perm: int = N_PERM) -> dict:
    """units[unit_id][site_id] = {"a": unit direction vector, "b": unit direction
    vector} -- half A/half B for the within-session tier, session A/session B
    for the across-session tier. Same-site cosine = cos(a, b) for one site;
    different-site cosine = cos(a of site i, b of site j) for i != j, within
    the same unit. Null: independently within each unit, permute which site
    label the "b" vectors carry (the fixed total per-unit cosine sum makes
    this an exact, cheap re-partition rather than a resampling loop)."""
    per_unit_matrix: dict[str, np.ndarray] = {}
    per_unit_paired_diff: dict[str, float] = {}
    same_vals, diff_vals = [], []
    for uid, sites in units.items():
        site_ids = sorted(sites)
        m = len(site_ids)
        if m < 2:
            continue
        A = np.stack([sites[s]["a"] for s in site_ids])
        B = np.stack([sites[s]["b"] for s in site_ids])
        cos_mat = (A @ B.T) / (
            np.linalg.norm(A, axis=1)[:, None] * np.linalg.norm(B, axis=1)[None, :] + 1e-12
        )
        per_unit_matrix[uid] = cos_mat
        same = np.diag(cos_mat)
        diff = cos_mat[~np.eye(m, dtype=bool)]
        same_vals.extend(same.tolist())
        diff_vals.extend(diff.tolist())
        per_unit_paired_diff[uid] = float(same.mean() - diff.mean())

    if not same_vals or not diff_vals:
        return {
            "status": BRANCH_INFEASIBLE,
            "branch": BRANCH_INFEASIBLE,
            "contrast": None,
            "n_units": len(per_unit_matrix),
            "reason": "fewer than two sites with a computable direction in every unit",
        }

    same_arr, diff_arr = np.array(same_vals), np.array(diff_vals)

    def _pooled_contrast(perms: dict[str, np.ndarray]) -> float:
        # Same summation path for the observed statistic (perms = identity)
        # and every null draw, so the identity permutation reproduces the
        # observed value bit-for-bit -- summing then dividing, rather than
        # np.mean's pairwise reduction, does not associate identically.
        same_sum = diff_sum = 0.0
        n_same = n_diff = 0
        for uid, perm in perms.items():
            cm = per_unit_matrix[uid]
            m = cm.shape[0]
            s = cm[np.arange(m), perm].sum()
            same_sum += s
            diff_sum += cm.sum() - s
            n_same += m
            n_diff += m * (m - 1)
        return same_sum / n_same - diff_sum / n_diff

    unit_ids = list(per_unit_matrix)
    identity_perms = {uid: np.arange(per_unit_matrix[uid].shape[0]) for uid in unit_ids}
    observed_contrast = _pooled_contrast(identity_perms)

    null = np.empty(n_perm)
    for p in range(n_perm):
        null[p] = _pooled_contrast({uid: rng.permutation(per_unit_matrix[uid].shape[0])
                                    for uid in unit_ids})

    p_value = permutation_pvalue(null >= observed_contrast)
    paired_diffs = list(per_unit_paired_diff.values())
    mdd = (minimum_detectable_paired_difference(paired_diffs) if len(paired_diffs) >= 2
           else {"status": "not_computable", "n": len(paired_diffs),
                 "reason": "fewer than 2 units contributed a same-vs-different contrast"})
    return {
        "status": "computed",
        "same_site_mean_cosine": float(same_arr.mean()),
        "different_site_mean_cosine": float(diff_arr.mean()),
        "contrast": observed_contrast,
        "n_same_site_cosines": int(len(same_arr)),
        "n_different_site_cosines": int(len(diff_arr)),
        "n_units": len(unit_ids),
        "p_value": p_value,
        "n_perm": n_perm,
        "minimum_detectable_difference": mdd,
        "mdd_reference": MDD_REFERENCE,
        "branch": _decision_branch(p_value, mdd),
    }


# ── Per-session assembly ───────────────────────────────────────────────────────

def _all_stim_conditions(session: dict) -> dict[str, list[int]]:
    """Every non-control condition SEEN in this session's own parameter
    table, regardless of whether its electrodes survived the shorted-channel
    exclusion -- the zero-drop ledger needs every seen site, not just the
    ones a downstream filter happens to keep."""
    control_idx = session["control_idx"]
    return {str(c): chan_ids for c, chan_ids in enumerate(session["stim_channels"])
            if c != control_idx and chan_ids}


def _channels_available(chan_ids: list[int], channel_ids: np.ndarray) -> bool:
    """Every stimulation electrode of this condition must survive the
    shorted-channel exclusion (the same requirement build_session_features
    applies before trusting a B) -- a partial electrode set misrepresents
    the delivered input."""
    available = set(channel_ids.tolist())
    return all(cid in available for cid in chan_ids)


def within_session_reliability_for_session(prefix: str, ledger: dict) -> dict | None:
    session = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    ledger["condition_cells"] += _condition_cell_ledger(
        prefix, _trials_by_cond_angle(session), session["control_idx"])
    fit = _fit_latent_basis(session, session["control_idx"])
    ledger["sessions"].append({
        "session": prefix, "status": "analysed" if fit is not None else "refused",
        "reason": None if fit is not None else
        f"fewer than {MIN_CONTROL_TRIALS_FOR_FIT} valid control trials to fit a latent basis",
    })
    if fit is None:
        return None
    V, var_explained, n_ctrl_fit = fit
    project = lambda v: v @ V

    control_idx = session["control_idx"]
    all_sites = _all_stim_conditions(session)
    angle_ids = sorted({a for (_c, a) in _trials_by_cond_angle(session)})
    table = _trials_by_cond_angle(session)
    half_a, half_b = _split_table_in_half(table)

    full_by_site, sd_scalar = {}, _control_residual_sd(table, control_idx, angle_ids, project)
    halves_by_site = {}
    for site_id, chan_ids in all_sites.items():
        if not _channels_available(chan_ids, session["channel_ids"]):
            ledger["within_session_sites"].append({
                "session": prefix, "site": site_id, "stim_channels": chan_ids,
                "status": "refused",
                "reason": f"stimulation electrode(s) {chan_ids} not present in this session's "
                          "recorded (post shorted-channel-exclusion) channel set",
            })
            continue
        site_cond = int(site_id)
        full = _site_displacement(table, control_idx, site_cond, angle_ids, project)
        ha = _site_displacement(half_a, control_idx, site_cond, angle_ids, project)
        hb = _site_displacement(half_b, control_idx, site_cond, angle_ids, project)
        full_by_site[site_id] = {
            "stim_channels": chan_ids, **full,
            "normalized_magnitude": (
                full["raw_magnitude"] / sd_scalar
                if full["status"] == "computed" and sd_scalar and sd_scalar > 1e-12 else None
            ),
        }
        both_ok = ha["status"] == "computed" and hb["status"] == "computed"
        reason = None if both_ok else (ha.get("reason") or hb.get("reason"))
        ledger["within_session_sites"].append({
            "session": prefix, "site": site_id, "stim_channels": chan_ids,
            "status": "analysed" if both_ok else "refused", "reason": reason,
        })
        if both_ok:
            halves_by_site[site_id] = {"a": ha["direction"], "b": hb["direction"]}

    return {
        "session": prefix, "var_explained": var_explained, "n_control_trials_fit": n_ctrl_fit,
        "n_pc": int(V.shape[1]), "control_residual_sd": sd_scalar,
        "site_displacements": full_by_site, "halves_for_reliability_test": halves_by_site,
    }


def across_session_reproducibility_for_pair(prefix_a: str, prefix_b: str, ledger: dict) -> dict:
    session_a = load_macaque_pfc_microstimulation_session(prefix_a, correct=True)
    session_b = load_macaque_pfc_microstimulation_session(prefix_b, correct=True)
    ids_a, ids_b = set(session_a["channel_ids"].tolist()), set(session_b["channel_ids"].tolist())
    shared = sorted(ids_a & ids_b)
    n_shared = len(shared)
    pair_record = {"session_a": prefix_a, "session_b": prefix_b, "n_shared_channels": n_shared,
                    "n_channels_a": len(ids_a), "n_channels_b": len(ids_b)}
    if n_shared < MIN_SHARED_CHANNELS:
        ledger["across_session_site_pairs"].append({
            "pair": [prefix_a, prefix_b], "site": None, "status": "refused",
            "reason": f"shared-channel basis has only {n_shared} channels, below the "
                      f"{MIN_SHARED_CHANNELS}-channel floor",
        })
        return {**pair_record, "status": BRANCH_INFEASIBLE,
                "reason": f"only {n_shared} shared channels (< {MIN_SHARED_CHANNELS})"}

    def _projector(session, idx_by_shared_pos):
        def project(v):
            return v[idx_by_shared_pos]
        return project

    idx_a = np.array([int(np.where(session_a["channel_ids"] == cid)[0][0]) for cid in shared])
    idx_b = np.array([int(np.where(session_b["channel_ids"] == cid)[0][0]) for cid in shared])
    project_a, project_b = _projector(session_a, idx_a), _projector(session_b, idx_b)

    table_a, table_b = _trials_by_cond_angle(session_a), _trials_by_cond_angle(session_b)
    angle_ids = sorted({a for (_c, a) in table_a} | {a for (_c, a) in table_b})
    sd_a = _control_residual_sd(table_a, session_a["control_idx"], angle_ids, project_a)
    sd_b = _control_residual_sd(table_b, session_b["control_idx"], angle_ids, project_b)

    # Match sites across the pair by their raw stimulation channel token
    # (identical physical electrode set), not by condition index (which the
    # loader assigns independently per file). Unfiltered by channel
    # availability -- a site excluded in only one session of the pair must
    # still appear in the ledger as a seen-but-refused site, not vanish.
    chan_to_id_a = {tuple(sorted(v)): k for k, v in _all_stim_conditions(session_a).items()}
    chan_to_id_b = {tuple(sorted(v)): k for k, v in _all_stim_conditions(session_b).items()}
    shared_site_channels = sorted(set(chan_to_id_a) & set(chan_to_id_b))

    displacements = {}
    sessions_for_reliability = {}
    for chans in shared_site_channels:
        site_label = "_".join(str(c) for c in chans)
        avail_a = _channels_available(list(chans), session_a["channel_ids"])
        avail_b = _channels_available(list(chans), session_b["channel_ids"])
        if not (avail_a and avail_b):
            missing = prefix_a if not avail_a else prefix_b
            ledger["across_session_site_pairs"].append({
                "pair": [prefix_a, prefix_b], "site": site_label, "status": "refused",
                "reason": f"stimulation electrode(s) {list(chans)} not present in {missing}'s "
                          "recorded (post shorted-channel-exclusion) channel set",
            })
            continue
        cond_a, cond_b = int(chan_to_id_a[chans]), int(chan_to_id_b[chans])
        disp_a = _site_displacement(table_a, session_a["control_idx"], cond_a, angle_ids, project_a)
        disp_b = _site_displacement(table_b, session_b["control_idx"], cond_b, angle_ids, project_b)
        both_ok = disp_a["status"] == "computed" and disp_b["status"] == "computed"
        reason = None if both_ok else (disp_a.get("reason") or disp_b.get("reason"))
        ledger["across_session_site_pairs"].append({
            "pair": [prefix_a, prefix_b], "site": site_label,
            "status": "analysed" if both_ok else "refused", "reason": reason,
        })
        displacements[site_label] = {
            "stim_channels": list(chans),
            "session_a": {**disp_a,
                          "normalized_magnitude": (disp_a["raw_magnitude"] / sd_a
                          if disp_a["status"] == "computed" and sd_a and sd_a > 1e-12 else None)},
            "session_b": {**disp_b,
                          "normalized_magnitude": (disp_b["raw_magnitude"] / sd_b
                          if disp_b["status"] == "computed" and sd_b and sd_b > 1e-12 else None)},
        }
        if both_ok:
            sessions_for_reliability[site_label] = {"a": disp_a["direction"], "b": disp_b["direction"]}

    return {**pair_record, "status": "computed", "control_residual_sd_a": sd_a,
            "control_residual_sd_b": sd_b, "site_displacements": displacements,
            "sessions_for_reliability_test": sessions_for_reliability}


def main():
    started = time.time()
    ledger = {"sessions": [], "within_session_sites": [], "condition_cells": [],
              "across_session_site_pairs": []}

    print("Within-session reliability -- fitting each session's own latent basis ...")
    per_session_within = {}
    for prefix in SESSIONS:
        res = within_session_reliability_for_session(prefix, ledger)
        if res is not None:
            per_session_within[prefix] = res
        print(f"  {prefix}: {'ok' if res is not None else 'REFUSED'}")

    within_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_site_reproducibility_within"))
    primary_units = {p: per_session_within[p]["halves_for_reliability_test"]
                      for p in PRIMARY_ANIMAL_SESSIONS if p in per_session_within}
    within_primary = site_identity_contrast(primary_units, within_rng)

    second_animal_units = ({SECOND_ANIMAL_SESSION: per_session_within[SECOND_ANIMAL_SESSION]["halves_for_reliability_test"]}
                            if SECOND_ANIMAL_SESSION in per_session_within else {})
    within_second_animal = site_identity_contrast(
        second_animal_units, np.random.default_rng(
            stable_seed("macaque_pfc_microstimulation_site_reproducibility_within_second_animal")))

    print("\nAcross-session reproducibility -- the three repeated channel sets ...")
    per_pair = {}
    for prefix_a, prefix_b in REPEAT_PAIRS:
        rec = across_session_reproducibility_for_pair(prefix_a, prefix_b, ledger)
        per_pair[f"{prefix_a}__{prefix_b}"] = rec
        print(f"  {prefix_a} vs {prefix_b}: n_shared_channels={rec['n_shared_channels']}, "
              f"status={rec['status']}")

    across_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_site_reproducibility_across"))
    across_units = {pid: rec["sessions_for_reliability_test"] for pid, rec in per_pair.items()
                    if rec["status"] == "computed"}
    across_result = site_identity_contrast(across_units, across_rng) if across_units else {
        "status": BRANCH_INFEASIBLE,
        "reason": "no repeat pair cleared the shared-channel/condition-coverage floor",
    }
    if across_result.get("status") not in ("computed",):
        across_result = {**across_result, "branch": BRANCH_INFEASIBLE}

    # ── Zero-drop assertions ───────────────────────────────────────────────
    def _assert_seen_eq_analysed_plus_refused(name, entries):
        analysed = sum(1 for e in entries if e["status"] == "analysed")
        refused = sum(1 for e in entries if e["status"] == "refused")
        assert len(entries) == analysed + refused, f"{name}: zero-drop violated"
        return {"seen": len(entries), "analysed": analysed, "refused": refused}

    zero_drop = {
        "sessions": _assert_seen_eq_analysed_plus_refused("sessions", ledger["sessions"]),
        "within_session_sites": _assert_seen_eq_analysed_plus_refused(
            "within_session_sites", ledger["within_session_sites"]),
        "across_session_site_pairs": _assert_seen_eq_analysed_plus_refused(
            "across_session_site_pairs", ledger["across_session_site_pairs"]),
        "condition_cells": _assert_seen_eq_analysed_plus_refused(
            "condition_cells", ledger["condition_cells"]),
    }

    wall_clock_s = round(time.time() - started, 3)
    out = {
        "scope": {
            "corpus": "macaque dorsolateral prefrontal microstimulation",
            "sessions_seen": zero_drop["sessions"]["seen"],
            "sessions_analysed": zero_drop["sessions"]["analysed"],
            "sessions_refused": zero_drop["sessions"]["refused"],
            "parameters": {
                "minimum_trials_per_arm": MIN_TRIALS_PER_ARM,
                "minimum_control_trials_for_fit": MIN_CONTROL_TRIALS_FOR_FIT,
                "minimum_shared_channels": MIN_SHARED_CHANNELS,
                "latent_rank": N_PC_LATENT,
                "permutations": N_PERM,
                "minimum_detectable_difference_reference": MDD_REFERENCE,
            },
            "seed": "deterministic stable_seed labels for each reliability tier",
            "wall_clock_s": wall_clock_s,
        },
        "code_commit": git_commit(ROOT),
        "wall_clock_s": wall_clock_s,
        "corpus_note": (
            "Amplitude is a single fixed value per session in this corpus (50 for the ten "
            "primary-animal sessions, 125 for the second animal's one session): this "
            "measurement can speak to where a site displaces the population state, never to "
            "how that displacement scales with stimulation intensity."
        ),
        "primary_animal_amplitude": 50,
        "second_animal_amplitude": 125,
        "zero_drop": zero_drop,
        "within_session_reliability": {
            "primary_animal": within_primary,
            "second_animal_session_reported_separately": within_second_animal,
        },
        "across_session_reproducibility": across_result,
        "per_session_site_displacements": {
            p: {"var_explained": v["var_explained"], "n_control_trials_fit": v["n_control_trials_fit"],
                "n_pc": v["n_pc"], "control_residual_sd": v["control_residual_sd"],
                "sites": {sid: {"stim_channels": s["stim_channels"], "status": s["status"],
                                "reason": s.get("reason"), "raw_magnitude": s.get("raw_magnitude"),
                                "normalized_magnitude": s.get("normalized_magnitude"),
                                "direction": s["direction"].tolist() if s.get("direction") is not None else None,
                                "n_angles_used": s.get("n_angles_used")}
                          for sid, s in v["site_displacements"].items()}}
            for p, v in per_session_within.items()
        },
        "repeat_pairs": {
            pid: {"session_a": rec["session_a"], "session_b": rec["session_b"],
                  "n_shared_channels": rec["n_shared_channels"], "status": rec["status"],
                  "reason": rec.get("reason"),
                  "site_displacements": {
                      sid: {"stim_channels": d["stim_channels"],
                            "session_a": {"status": d["session_a"]["status"],
                                          "reason": d["session_a"].get("reason"),
                                          "raw_magnitude": d["session_a"].get("raw_magnitude"),
                                          "normalized_magnitude": d["session_a"].get("normalized_magnitude"),
                                          "direction": (d["session_a"]["direction"].tolist()
                                                        if d["session_a"].get("direction") is not None else None)},
                            "session_b": {"status": d["session_b"]["status"],
                                          "reason": d["session_b"].get("reason"),
                                          "raw_magnitude": d["session_b"].get("raw_magnitude"),
                                          "normalized_magnitude": d["session_b"].get("normalized_magnitude"),
                                          "direction": (d["session_b"]["direction"].tolist()
                                                        if d["session_b"].get("direction") is not None else None)}}
                      for sid, d in rec.get("site_displacements", {}).items()}}
            for pid, rec in per_pair.items()
        },
        "ledger_detail": ledger,
    }

    print(f"\nWithin-session (primary animal, n={within_primary.get('n_units')} sessions): "
          f"same={within_primary.get('same_site_mean_cosine')}, "
          f"diff={within_primary.get('different_site_mean_cosine')}, "
          f"contrast={within_primary.get('contrast')}, p={within_primary.get('p_value')}, "
          f"branch={within_primary.get('branch')}")
    print(f"Within-session (second animal, its own session): "
          f"same={within_second_animal.get('same_site_mean_cosine')}, "
          f"diff={within_second_animal.get('different_site_mean_cosine')}, "
          f"branch={within_second_animal.get('branch')}")
    print(f"Across-session (n_units={across_result.get('n_units')} repeat pairs): "
          f"same={across_result.get('same_site_mean_cosine')}, "
          f"diff={across_result.get('different_site_mean_cosine')}, "
          f"contrast={across_result.get('contrast')}, p={across_result.get('p_value')}, "
          f"branch={across_result.get('branch')}")

    out_path = RESULTS / "macaque_pfc_microstimulation_site_reproducibility.json"
    with open(out_path, "w") as f:
        import json
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["macaque_pfc_microstimulation_site_reproducibility"] = _json_safe(out)
    print(f"\nSaved {out_path}, updated all_statistics.json")


if __name__ == "__main__":
    main()
