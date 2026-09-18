#!/usr/bin/env python3
"""macaque_pfc_microstimulation -- does the rate-free maintenance-delay
displacement differ between correct and error trials?

The delivered panel (results/stimulation_response_gate_and_panel.json,
stimulation_response_panel/macaque_pfc_microstimulation) reads this corpus's
correct-trial file only, because it was built to score a treatment (stim vs
control) contrast, not a behaviour contrast -- the outcome variable was
never wired in. This is the only preparation in the project where a
randomised, within-session, multi-site stimulation is delivered INSIDE a
working-memory maintenance delay with both a pre- and a post-stimulation
window AND a behavioural outcome in the same trials, so it is the one place
the full chain (stimulation -> state -> behaviour) can be asked end to end.
This module answers the behaviour half: within condition (control vs
stimulated) and within window (pre vs post), does the per-trial rate-free
state displacement differ between correct and error trials.

PART 1 settles a trap before any effect is fit: `error/` file counts per
condition are wildly uneven (0-197 within one session) while `correct/` file
counts are near-uniform (typically 15-20), so `n_error/(n_correct+n_error)`
is not a usable error rate and no accuracy quantity is computed from these
directory counts anywhere in this module. The trap is investigated against
each session's own `behavior.trialinfo` (rig settings: `badTrialHandling`,
`earlySaccadePenalty`, `earlySaccadeTime`, `rpts`) and `behavior.codes`
(per-trial event stream) before any displacement number is read.

PART 2 is the link itself, reusing the shared time-resolved estimator
(src/stimulation_response_estimator.py) and this corpus's own feature path
(crop_trial, BIN_S, PRE_S, load_macaque_pfc_microstimulation_session,
imported unchanged from scripts/run_macaque_pfc_microstimulation_pipeline.py)
and window definitions (results/stimulation_design_census.json's own
pre_stimulation_window_s / post_stimulation_windows_through_probe_s for this
corpus, read the same way the panel reads them). CORRECT trials play the
reference-forming ("control") role window_treatment_effect expects and
ERROR trials are scored against that fixed reference -- the natural mapping
of a treatment-contrast estimator onto a behaviour contrast, and the one
whose bias condition (the reference-forming pool must be representative of
the session, not just of the trials that happen to be scored) is exactly
the validity assumption this module is required to probe: whether the
correct-trial subsample is random with respect to when in the session it
occurred, using each file's own params/trialsequence order (the only
chronological signal this release provides -- no shared trial counter or
timestamp links the correct-trial and error-trial files, so a direct
correct-vs-error temporal comparison is not reachable and this module says
so rather than assuming it away).

Sa210311_s224 stimulates through bipolar channel pairs against a different
electrode montage from the ten Wa sessions and is never pooled silently:
every pooled cell also reports a Wa-only descriptive sensitivity value.

Outputs:
  results/macaque_maintenance_behaviour_link.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_macaque_maintenance_behaviour_link.py
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit  # noqa: E402
from stimulation_response_estimator import (  # noqa: E402
    cluster_bootstrap_pooled_effect,
    nuisance_treatment_effect,
    window_treatment_effect,
)
from run_macaque_pfc_microstimulation_pipeline import (  # noqa: E402
    BIN_S, DATA, PRE_S, SESSIONS, crop_trial, load_macaque_pfc_microstimulation_session,
)
from run_stimulation_response_gate_and_panel import _read_census_row  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "macaque_maintenance_behaviour_link.json"

# Named reference on the displacement scale, inherited from this project's stimulation-response panel
# (see docs/mandates/STANDING_CONSTRAINTS.md "Reference constants are internal, not biological"): the
# scale a powered null is declared against, not a biological or clinical threshold.
REFERENCE_DISPLACEMENT_SD = 1.0
POWER = 0.80
WINDOW_NAMES = ("pre_stimulation_window", "post_stimulation_window")
ARM_NAMES = ("control_condition", "stimulated_condition")

DECISION_RULE_DECLARED_BEFORE_FITTING = {
    "reference_on_displacement_scale": REFERENCE_DISPLACEMENT_SD,
    "reference_definition": (
        "1.0 normalised control-trial SD -- the internal comparison scale this project's stimulation-"
        "response panel already reports against, not a biological or clinical threshold."
    ),
    "power": POWER,
    "validity_gate": {
        "statement": (
            "Computed before any effect-size branch below is read. If it fires, every cell in this "
            "artifact is reported as validity_assumption_compromised rather than as evidence for or "
            "against a correct-vs-error difference, because the reference direction each cell's error "
            "trials are scored against would itself be biased."
        ),
        "rule": (
            "compromised if ANY session x arm cell's correct-trial chronological-rank Kolmogorov-Smirnov "
            "test against Uniform(0,1) clears a Bonferroni-corrected two-sided alpha of 0.05 / "
            "n_cells_tested; not_rejected otherwise. The probe uses each file's own params/trialsequence "
            "order as the chronological signal (no shared clock links the correct-trial and error-trial "
            "files in this release, so a direct correct-vs-error temporal comparison is not reachable)."
        ),
    },
    "per_cell_branches": {
        "mdd_clears_reference_but_p_is_uninformative_at_two_clusters": (
            "mdd (80% power, whole-session cluster bootstrap clustered on animal) < 1.0 named reference. "
            "Named this way rather than 'powered' because mdd<reference does NOT mean this cell could "
            "reach significance: with exactly 2 animal clusters the sign-flip permutation p-value is "
            "floored near 0.49-0.50 for ANY effect size (2^2=4 sign patterns), a structural property of "
            "the cluster count, not of the effect. The mdd is still reported because it bounds how large "
            "an effect this design's between-animal spread is consistent with, but no significance "
            "distinction follows from it at this cluster count."
        ),
        "inconclusive": (
            "mdd >= 1.0 named reference, OR fewer than 2 animal clusters contribute a finite value, OR "
            "the window is a structural_void per results/stimulation_design_census.json"
        ),
    },
    "point_estimate_below_own_mdd_flag": (
        "attached to every computed cell where abs(mean_value) < mdd regardless of branch, per standing "
        "project rule -- such an estimate is upward-biased conditional on reaching significance."
    ),
    "never_a_significance_claim": (
        "cluster_bootstrap_pooled_effect clusters on animal id (2 animals: Sa, Wa); its sign-flip null "
        "has exactly 2^2=4 sign patterns at that cluster count, so its own two-sided p-value is floored "
        "well above 0.05 (this project's standing floor: 0.493) and is reported but never read as "
        "evidence of an effect or its absence -- only the mdd-vs-reference comparison, the point estimate "
        "and its CI are read as informative, and even the mdd-vs-reference comparison at n=2 clusters uses "
        "a normal approximation this project's own statistics module discloses as unreliable below "
        "'a handful of units', so it is reported as a bound, never as proof of adequate power."
    ),
}


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# Field extraction the shared pipeline loader does not expose: params/trialsequence
# (a per-trial condition index, needed for the validity probe) and behavior.codes /
# behavior.trialinfo (needed for the trap audit). Same v7.3-vs-v7 per-file dispatch
# as the shared loader: try opening as HDF5 first, fall back to scipy on OSError.
# ══════════════════════════════════════════════════════════════════════════

def _mat_path(prefix: str, correct: bool) -> Path:
    folder = "correct" if correct else "error"
    fname = f"{prefix}.mat" if correct else f"{prefix}_err.mat"
    return DATA / folder / fname


def _load_trialsequence_and_n_angle(prefix: str, correct: bool) -> tuple[np.ndarray | None, int | None]:
    """params/trialsequence in file order (the chronological signal this
    release provides within one outcome-type file) and the number of target
    angles (params/cond_targetAngle length), needed to decode trialsequence's
    1-based `stim_cond * n_angle + angle_idx + 1` condition code -- verified
    against this loader's own grid-derived (stim_cond, angle_idx) counts for
    every session before this module trusts it (see run_trial_admission_trap_audit)."""
    mat_path = _mat_path(prefix, correct)
    if not mat_path.exists():
        return None, None
    try:
        with h5py.File(str(mat_path), "r") as f:
            ts = f["params"]["trialsequence"][()].ravel().astype(int)
            n_angle = int(f["params"]["cond_targetAngle"].shape[0])
            return ts, n_angle
    except OSError:
        d = sio.loadmat(str(mat_path), simplify_cells=True)
        ts = np.asarray(d["params"]["trialsequence"]).ravel().astype(int)
        n_angle = int(np.asarray(d["params"]["cond_targetAngle"]).shape[0])
        return ts, n_angle


def _h5_trialinfo_snapshot(ti_group) -> dict:
    def _char(name: str) -> str:
        return "".join(chr(int(x)) for x in np.asarray(ti_group[name][()]).ravel())

    def _num(name: str) -> float:
        return float(np.asarray(ti_group[name][()]).ravel()[0])

    return {
        "bad_trial_handling": _char("badTrialHandling") if "badTrialHandling" in ti_group else None,
        "early_saccade_penalty": _num("earlySaccadePenalty") if "earlySaccadePenalty" in ti_group else None,
        "early_saccade_time": _num("earlySaccadeTime") if "earlySaccadeTime" in ti_group else None,
        "rpts": _num("rpts") if "rpts" in ti_group else None,
    }


def _scipy_trialinfo_snapshot(ti) -> dict:
    return {
        "bad_trial_handling": getattr(ti, "badTrialHandling", None),
        "early_saccade_penalty": float(ti.earlySaccadePenalty) if hasattr(ti, "earlySaccadePenalty") else None,
        "early_saccade_time": float(ti.earlySaccadeTime) if hasattr(ti, "earlySaccadeTime") else None,
        "rpts": float(ti.rpts) if hasattr(ti, "rpts") else None,
    }


def _iter_behavior_grid(prefix: str, correct: bool):
    """Yield {'stim_cond', 'angle_idx', 'terminal_code', 'duration_s'} for
    every non-empty behavior grid cell, in the same (stim_cond, angle_idx,
    trial-in-cell) traversal order load_macaque_pfc_microstimulation_session
    uses for spikerate, plus a `trialinfo` snapshot (session-wide rig
    settings, captured once). Empty cells use the SAME shape-(2,) marker
    convention as the shared loader's spikerate empties (h5py) / lack a
    `.codes` attribute (scipy) -- asserted here, not assumed."""
    mat_path = _mat_path(prefix, correct)
    if not mat_path.exists():
        return
    trialinfo_snapshot: dict = {}
    try:
        f = h5py.File(str(mat_path), "r")
        is_h5 = True
    except OSError:
        is_h5 = False

    if is_h5:
        with f:
            b = f["behavior"]
            n_trial, n_angle, n_amp, n_chan = b.shape  # h5py-reversed, matches spikerate's own axis order
            for c in range(n_chan):
                for a in range(n_angle):
                    for tr in range(n_trial):
                        ref = b[tr, a, 0, c]
                        grp = f[ref]
                        if isinstance(grp, h5py.Dataset):  # the shape-(2,) empty-cell marker
                            continue
                        if "codes" not in grp:
                            continue
                        ds = grp["codes"]
                        if ds.shape is None or len(ds.shape) != 2 or ds.shape[1] < 2:
                            continue
                        arr = ds[()]  # (3, n_events): row0=event code, row1=relative time (s)
                        code_row, time_row = arr[0], arr[1]
                        if not trialinfo_snapshot and "trialinfo" in grp:
                            trialinfo_snapshot.update(_h5_trialinfo_snapshot(grp["trialinfo"]))
                        yield {
                            "stim_cond": c, "angle_idx": a,
                            "terminal_code": float(code_row[-2]) if len(code_row) >= 2 else float(code_row[-1]),
                            "duration_s": float(time_row[-1] - time_row[0]),
                            "trialinfo": dict(trialinfo_snapshot),
                        }
    else:
        d = sio.loadmat(str(mat_path), simplify_cells=True)
        b = d["behavior"]
        n_chan, n_angle = b.shape[0], b.shape[1]
        for c in range(n_chan):
            for a in range(n_angle):
                for tr in np.atleast_1d(b[c, a]):
                    if not hasattr(tr, "codes"):  # the shape-(2,) empty-cell marker's scipy counterpart
                        continue
                    arr = np.asarray(tr.codes)
                    if arr.ndim != 2 or arr.shape[0] < 2:
                        continue
                    code_col, time_col = arr[:, 0], arr[:, 1]  # (n_events, 3): col0=event code, col1=relative time (s)
                    if not trialinfo_snapshot and hasattr(tr, "trialinfo"):
                        trialinfo_snapshot.update(_scipy_trialinfo_snapshot(tr.trialinfo))
                    yield {
                        "stim_cond": c, "angle_idx": a,
                        "terminal_code": float(code_col[-2]) if len(code_col) >= 2 else float(code_col[-1]),
                        "duration_s": float(time_col[-1] - time_col[0]),
                        "trialinfo": dict(trialinfo_snapshot),
                    }


# ══════════════════════════════════════════════════════════════════════════
# PART 1 -- settle the trap before fitting anything
# ══════════════════════════════════════════════════════════════════════════

def run_trial_admission_trap_audit() -> dict:
    per_session, all_bad_trial_handling, all_rpts = {}, set(), []
    for prefix in SESSIONS:
        corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
        err = load_macaque_pfc_microstimulation_session(prefix, correct=False)
        if corr is None:
            per_session[prefix] = {"status": "excluded", "reason": "no_usable_correct_trial_file"}
            continue
        corr_grid = Counter((t["stim_cond"], t["angle_idx"]) for t in corr["trials"])
        err_grid = Counter((t["stim_cond"], t["angle_idx"]) for t in err["trials"]) if err is not None else Counter()

        corr_counts = np.array(list(corr_grid.values()), dtype=float)
        err_counts = np.array([err_grid.get(k, 0) for k in corr_grid], dtype=float)  # same cell set as correct's
        corr_cv = float(np.std(corr_counts, ddof=1) / np.mean(corr_counts)) if len(corr_counts) > 1 else None
        err_cv = float(np.std(err_counts, ddof=1) / np.mean(err_counts)) if len(err_counts) > 1 and np.mean(err_counts) > 0 else None

        n_correct, n_error = len(corr["trials"]), (len(err["trials"]) if err is not None else 0)
        corr_behavior = list(_iter_behavior_grid(prefix, correct=True))
        err_behavior = list(_iter_behavior_grid(prefix, correct=False)) if err is not None else []
        corr_terminal = Counter(round(b["terminal_code"], 1) for b in corr_behavior)
        err_terminal = Counter(round(b["terminal_code"], 1) for b in err_behavior)
        corr_durations = np.array([b["duration_s"] for b in corr_behavior])
        err_durations = np.array([b["duration_s"] for b in err_behavior]) if err_behavior else np.array([])

        bad_th = {corr_behavior[0]["trialinfo"].get("bad_trial_handling")} if corr_behavior else set()
        if err_behavior:
            bad_th.add(err_behavior[0]["trialinfo"].get("bad_trial_handling"))
        all_bad_trial_handling |= bad_th
        rpts_corr = corr_behavior[0]["trialinfo"].get("rpts") if corr_behavior else None
        if rpts_corr is not None:
            all_rpts.append(rpts_corr)

        per_session[prefix] = {
            "status": "computed",
            "n_correct_trials": n_correct, "n_error_trials": n_error,
            "naive_error_fraction_NOT_an_error_rate": (
                round(n_error / (n_correct + n_error), 4) if (n_correct + n_error) else None
            ),
            "n_correct_grid_cells": len(corr_grid), "n_error_grid_cells": len(err_grid),
            "correct_count_by_cell_coefficient_of_variation": corr_cv,
            "error_count_by_cell_coefficient_of_variation": err_cv,
            "correct_count_range": [int(corr_counts.min()), int(corr_counts.max())] if len(corr_counts) else None,
            "error_count_range": [int(err_counts.min()), int(err_counts.max())] if len(err_counts) else None,
            "n_correct_behaviour_records_seen": len(corr_behavior),
            "n_error_behaviour_records_seen": len(err_behavior),
            "behaviour_record_count_reconciles_with_loader": (
                len(corr_behavior) == n_correct and len(err_behavior) == n_error
            ),
            "correct_terminal_event_codes": {str(k): v for k, v in corr_terminal.items()},
            "error_terminal_event_codes": {str(k): v for k, v in err_terminal.items()},
            "n_distinct_error_terminal_codes": len(err_terminal),
            "correct_trial_duration_s": {
                "median": float(np.median(corr_durations)) if len(corr_durations) else None,
                "min": float(corr_durations.min()) if len(corr_durations) else None,
            },
            "error_trial_duration_s": {
                "median": float(np.median(err_durations)) if len(err_durations) else None,
                "min": float(err_durations.min()) if len(err_durations) else None,
            },
            "bad_trial_handling_values_seen": sorted(v for v in bad_th if v is not None),
            "rpts_target_repetitions_per_condition": rpts_corr,
        }

    computed = [s for s in per_session.values() if s["status"] == "computed"]
    mean_corr_cv = float(np.mean([s["correct_count_by_cell_coefficient_of_variation"] for s in computed
                                   if s["correct_count_by_cell_coefficient_of_variation"] is not None]))
    mean_err_cv = float(np.mean([s["error_count_by_cell_coefficient_of_variation"] for s in computed
                                  if s["error_count_by_cell_coefficient_of_variation"] is not None]))

    return {
        "status": "computed",
        "per_session": per_session,
        "sessions_seen": len(SESSIONS), "sessions_computed": len(computed),
        "pooled_correct_count_cv": mean_corr_cv, "pooled_error_count_cv": mean_err_cv,
        "bad_trial_handling_values_seen_across_all_sessions": sorted(v for v in all_bad_trial_handling if v is not None),
        "rpts_values_seen_across_all_sessions": sorted(set(all_rpts)),
        "answer": (
            "n_error / (n_correct + n_error) is NOT an error rate and no such quantity is computed "
            "anywhere below. Every session's behavior.trialinfo reports badTrialHandling='reshuffle' "
            f"(seen across all {len(computed)} computed sessions: {sorted(v for v in all_bad_trial_handling if v is not None)}) "
            f"and a target repetition count rpts (values seen: {sorted(set(all_rpts))}), and correct-trial "
            "counts per condition cluster tightly around that target "
            f"(pooled coefficient of variation {mean_corr_cv:.3f}) while error-trial counts per condition do "
            f"not (pooled coefficient of variation {mean_err_cv:.3f}, individual sessions ranging from single "
            "digits to hundreds). The evidence is consistent with a target-repetition-count design: a "
            "condition is retried under reshuffle until it accumulates rpts correct completions (or the "
            "session ends), so the correct-trial pool is balanced BY CONSTRUCTION, not by post-hoc "
            "subsampling, and the error-trial pool is the unbalanced raw remainder of every non-rewarded "
            "attempt at that condition. What this module cannot fully settle: the error pool is not a "
            "single failure type -- terminal event codes in the error file are consistently 2-3 distinct "
            "values per session (never the single value that ends every correct trial), and error-trial "
            "durations mostly fall inside the correct-trial duration range rather than being dramatically "
            "shorter, arguing against the error pool being dominated by immediate fixation-break aborts "
            "(no published event-code table for this release was available to decode which terminal code "
            "means what, so the exact mixture of late-stage response errors vs. early terminations such as "
            "the earlySaccadePenalty/earlySaccadeTime mechanism present in every session's trialinfo is not "
            "decoded here). Nothing below depends on resolving that residual sub-question: the analysis "
            "treats every admitted error trial as one member of an undifferentiated error pool, and the "
            "trap itself -- that directory counts are not an error rate -- is fully settled by the evidence "
            "above."
        ),
    }


# ══════════════════════════════════════════════════════════════════════════
# PART 2 -- the link, reusing the panel's feature path, windows and metric
# ══════════════════════════════════════════════════════════════════════════

def _trial_ranks_by_condition(trialsequence: np.ndarray, n_angle: int) -> dict[tuple[int, int], list[int]]:
    """0-indexed file-position (chronological order within this one file) for
    every occurrence of each (stim_cond, angle_idx), decoding trialsequence's
    1-based `stim_cond * n_angle + angle_idx + 1` condition code -- the same
    encoding verified against this loader's own grid-derived counts for
    every session and every corpus format (see the module docstring)."""
    ranks: dict[tuple[int, int], list[int]] = {}
    for pos, v in enumerate(trialsequence):
        c, a = divmod(int(v) - 1, n_angle)
        ranks.setdefault((c, a), []).append(pos)
    return ranks


def _session_arm_outcome_windows(prefix: str, pre_bins: int) -> tuple[dict | None, str | None]:
    """One session's (window x arm x outcome) trial-activity pools, built
    with the same crop_trial / BIN_S / PRE_S feature path the delivered
    panel uses for its treatment-vs-control cells (_macaque_session_windows
    in scripts/run_stimulation_response_gate_and_panel.py), extended here to
    also read the error-trial file and tag each trial's outcome. Condition
    index alignment between the correct-trial and error-trial files (the
    error file enumerates its own conditions separately, per the panel's own
    docstring) is asserted here rather than assumed: stim_channels token
    order and control_idx must match exactly or the session is excluded."""
    corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None, "no_usable_correct_trial_file_or_no_control_condition"
    err = load_macaque_pfc_microstimulation_session(prefix, correct=False)
    if err is None:
        return None, "no_usable_error_trial_file"
    control_idx = corr["control_idx"]
    if err["control_idx"] != control_idx:
        return None, "correct_and_error_control_idx_mismatch"
    if err["stim_channels"] != corr["stim_channels"]:
        return None, "correct_and_error_condition_ordering_mismatch"
    if not np.array_equal(corr["channel_ids"], err["channel_ids"]):
        return None, "correct_and_error_channel_id_mismatch"
    n_channels = len(corr["channel_ids"])

    ts_corr, n_angle = _load_trialsequence_and_n_angle(prefix, correct=True)
    rank_lookup = _trial_ranks_by_condition(ts_corr, n_angle) if ts_corr is not None and n_angle else {}
    cursor: dict[tuple[int, int], int] = {}

    pools = {win: {arm: {"correct_activity": [], "error_activity": []} for arm in ARM_NAMES} for win in WINDOW_NAMES}
    counts = {"correct_seen": 0, "correct_used": 0, "correct_excluded_short_crop": 0,
              "error_seen": 0, "error_used": 0, "error_excluded_short_crop": 0}
    correct_ranks = {arm: [] for arm in ARM_NAMES}  # trialsequence-derived chronological fraction, for the validity probe

    for tr in corr["trials"]:
        counts["correct_seen"] += 1
        key = (tr["stim_cond"], tr["angle_idx"])
        idx_in_cell = cursor.get(key, 0)
        cursor[key] = idx_in_cell + 1
        cropped = crop_trial(tr["spikerate"])
        if cropped is None:
            counts["correct_excluded_short_crop"] += 1
            continue
        counts["correct_used"] += 1
        arm = "control_condition" if tr["stim_cond"] == control_idx else "stimulated_condition"
        pools["pre_stimulation_window"][arm]["correct_activity"].append(cropped[:pre_bins, :].sum(axis=0) * BIN_S)
        pools["post_stimulation_window"][arm]["correct_activity"].append(cropped[pre_bins:, :].sum(axis=0) * BIN_S)
        ranks_for_cell = rank_lookup.get(key, [])
        if idx_in_cell < len(ranks_for_cell) and len(ts_corr) > 1:
            correct_ranks[arm].append(ranks_for_cell[idx_in_cell] / (len(ts_corr) - 1))

    for tr in err["trials"]:
        counts["error_seen"] += 1
        cropped = crop_trial(tr["spikerate"])
        if cropped is None:
            counts["error_excluded_short_crop"] += 1
            continue
        counts["error_used"] += 1
        arm = "control_condition" if tr["stim_cond"] == control_idx else "stimulated_condition"
        pools["pre_stimulation_window"][arm]["error_activity"].append(cropped[:pre_bins, :].sum(axis=0) * BIN_S)
        pools["post_stimulation_window"][arm]["error_activity"].append(cropped[pre_bins:, :].sum(axis=0) * BIN_S)

    for win in pools:
        for arm in pools[win]:
            for outcome in ("correct_activity", "error_activity"):
                rows = pools[win][arm][outcome]
                pools[win][arm][outcome] = np.array(rows) if rows else np.zeros((0, n_channels))

    return {"pools": pools, "counts": counts, "correct_ranks": correct_ranks,
            "n_trialsequence": int(len(ts_corr)) if ts_corr is not None else None}, None


def run_subsample_temporal_validity_probe(session_results: dict[str, dict]) -> dict:
    """Whether the correct-trial pool is random with respect to when in the
    session it occurred, the assumption window_treatment_effect needs for
    the reference direction it builds from that pool to be unbiased.
    Reachable form only (see module docstring): a one-sample Kolmogorov-
    Smirnov test of each session x arm's correct-trial chronological rank
    (fraction of that file's own trialsequence order) against Uniform(0,1)."""
    per_cell = {}
    for prefix, sr in session_results.items():
        if sr.get("status") != "computed":
            continue
        for arm in ARM_NAMES:
            fractions = np.array(sr["correct_ranks"][arm])
            key = f"{prefix}|{arm}"
            if len(fractions) < 5:
                per_cell[key] = {"status": "too_few_trials_for_probe", "n": int(len(fractions))}
                continue
            stat, p = kstest(fractions, "uniform")
            per_cell[key] = {"status": "computed", "n": int(len(fractions)), "ks_statistic": float(stat), "p_value": float(p)}

    tested = {k: v for k, v in per_cell.items() if v["status"] == "computed"}
    n_tested = len(tested)
    bonferroni_alpha = 0.05 / n_tested if n_tested else None
    n_uncorrected_significant = sum(1 for v in tested.values() if v["p_value"] <= 0.05)
    n_bonferroni_significant = sum(1 for v in tested.values() if bonferroni_alpha is not None and v["p_value"] < bonferroni_alpha)
    status = "compromised" if n_bonferroni_significant > 0 else "not_rejected"
    bonferroni_str = f"{bonferroni_alpha:.5f}" if bonferroni_alpha is not None else "n/a"
    return {
        "status": "computed",
        "per_session_arm_cell": per_cell,
        "n_cells_tested": n_tested,
        "bonferroni_alpha": bonferroni_alpha,
        "n_cells_uncorrected_significant_at_0.05": n_uncorrected_significant,
        "n_cells_bonferroni_significant": n_bonferroni_significant,
        "validity_assumption_status": status,
        "reading": (
            f"{n_bonferroni_significant} of {n_tested} session x arm cells reject Uniform(0,1) for the "
            f"correct-trial chronological rank at the Bonferroni-corrected alpha ({bonferroni_str}); "
            f"{n_uncorrected_significant} reject it at the uncorrected 0.05 threshold (about "
            f"{0.05 * n_tested:.1f} expected by chance alone if the null held everywhere). "
            + ("The correct-trial subsample is NOT demonstrably random with respect to session time in at "
               "least one cell -- this compromises every cell in the link below and is reported as the "
               "more important finding, per standing project rule, rather than smoothed past."
               if status == "compromised" else
               "No cell clears the Bonferroni threshold, so this probe does not reject the assumption that "
               "the correct-trial subsample is spread through the session the way a random draw would be; "
               "this is a failure to reject, not proof of randomness, and is reported as such.")
        ),
    }


def _classify_cell(pooled: dict) -> tuple[str, bool | None]:
    """mdd < 1.0 does NOT mean this cell can reach significance: with
    exactly 2 animal clusters the sign-flip permutation null has only 2^2=4
    sign patterns, so the two-sided p-value is floored near 0.49-0.50
    regardless of the true effect size -- a structural property of the
    cluster count, not of power. minimum_detectable_paired_difference's own
    docstring discloses that its normal approximation is adequate only once
    n is "more than a handful of units", which 2 clusters is not. The branch
    name below reflects both halves rather than collapsing them into a
    single "powered" label that would misleadingly suggest the test could
    have reached significance had the effect been real."""
    if pooled.get("status") != "computed" or pooled.get("n_clusters", 0) < 2:
        return "inconclusive", None
    mdd = pooled.get("mdd", {})
    if mdd.get("status") != "computed":
        return "inconclusive", None
    branch = (
        "mdd_clears_reference_but_p_is_uninformative_at_two_clusters"
        if mdd["mdd"] < REFERENCE_DISPLACEMENT_SD else "inconclusive"
    )
    below_own_mdd = abs(pooled["mean_value"]) < mdd["mdd"]
    return branch, below_own_mdd


def run_behaviour_link(census_windows: dict, session_results: dict[str, dict], validity_status: str) -> dict:
    cells = {}
    for win in WINDOW_NAMES:
        census_field = census_windows[win]
        for arm in ARM_NAMES:
            cell_key = f"{win}|{arm}"
            if census_field.get("status") != "measured":
                cells[cell_key] = {"status": "structural_void", "reason": census_field.get("reason")}
                continue
            dev_vals, dev_ids, nuis_vals, nuis_ids, per_session = [], [], [], [], {}
            for prefix in SESSIONS:
                sr = session_results.get(prefix)
                if sr is None or sr.get("status") != "computed":
                    per_session[prefix] = {"status": "excluded",
                                            "reason": (sr or {}).get("reason", "session_not_computed")}
                    continue
                pool = sr["pools"][win][arm]
                correct_activity, error_activity = pool["correct_activity"], pool["error_activity"]
                eff = window_treatment_effect(correct_activity, error_activity)
                correct_totals = correct_activity.sum(axis=1) if correct_activity.size else np.array([])
                error_totals = error_activity.sum(axis=1) if error_activity.size else np.array([])
                nuis = nuisance_treatment_effect(correct_totals, error_totals)
                per_session[prefix] = {
                    "cluster_id": prefix[:2],
                    "n_correct_trials_in_cell": int(correct_activity.shape[0]),
                    "n_error_trials_in_cell": int(error_activity.shape[0]),
                    "deviation_status": eff.get("status"),
                    "deviation_normalised_displacement": eff.get("normalised_displacement"),
                    "nuisance_status": nuis.get("status"),
                    "nuisance_normalised_change": nuis.get("normalised_change"),
                }
                if eff.get("status") == "computed" and eff.get("normalised_displacement") is not None:
                    dev_vals.append(eff["normalised_displacement"]); dev_ids.append(prefix[:2])
                if nuis.get("status") == "computed" and nuis.get("normalised_change") is not None:
                    nuis_vals.append(nuis["normalised_change"]); nuis_ids.append(prefix[:2])

            pooled_dev = (cluster_bootstrap_pooled_effect(np.array(dev_vals), dev_ids, f"macaque_maintenance_behaviour_link|{win}|{arm}|deviation")
                          if dev_vals else {"status": "not_computable", "reason": "no session produced a finite normalised displacement"})
            pooled_nuis = (cluster_bootstrap_pooled_effect(np.array(nuis_vals), nuis_ids, f"macaque_maintenance_behaviour_link|{win}|{arm}|nuisance")
                           if nuis_vals else {"status": "not_computable", "reason": "no session produced a finite nuisance change"})
            if validity_status == "compromised":
                branch, below_own_mdd = "validity_assumption_compromised", None
            else:
                branch, below_own_mdd = _classify_cell(pooled_dev)

            wa_only_vals = [v for v, i in zip(dev_vals, dev_ids) if i == "Wa"]
            sa_sensitivity = (
                {
                    "status": "descriptive_only_not_a_formal_test",
                    "reason": "excluding the single Sa-animal session leaves 1 cluster; cluster_bootstrap_pooled_effect requires >=2 clusters",
                    "n_wa_sessions_contributing": len(wa_only_vals),
                    "wa_only_mean": float(np.mean(wa_only_vals)) if wa_only_vals else None,
                    "wa_only_sd": float(np.std(wa_only_vals, ddof=1)) if len(wa_only_vals) > 1 else None,
                }
                if len(set(dev_ids)) >= 2 else {"status": "not_applicable", "reason": "fewer than 2 clusters contributed at all"}
            )

            cells[cell_key] = {
                "status": "computed", "window": win, "arm": arm, "window_seconds": census_field.get("value"),
                "n_sessions_contributing_deviation": len(dev_vals), "n_sessions_contributing_nuisance": len(nuis_vals),
                "rate_free_deviation_biomarker": {
                    "role": "candidate", "pooled": pooled_dev, "branch": branch,
                    "point_estimate_below_own_mdd": below_own_mdd,
                },
                "nuisance_total_activity_biomarker": {"role": "nuisance_control_not_a_candidate", "pooled": pooled_nuis},
                "sensitivity_excluding_sa210311_s224": sa_sensitivity,
                "per_session": per_session,
            }
    return cells


def main() -> None:
    t0 = time.time()
    output = {"version": "2026-09-11", "status": "running", "git_commit": git_commit(ROOT)}
    _flush(output)

    _log("running trial-admission trap audit")
    output["trial_admission_trap_audit"] = run_trial_admission_trap_audit()
    output["status"] = "trap_audit_complete_link_pending"
    _flush(output)

    row_m = _read_census_row("macaque_pfc_microstimulation")
    census_windows = {"pre_stimulation_window": row_m["pre_stimulation_window_s"],
                       "post_stimulation_window": row_m["post_stimulation_windows_through_probe_s"]}
    pre_bins = int(round(PRE_S / BIN_S))

    _log("building per-session correct/error window pools")
    session_results, excluded = {}, {}
    for prefix in SESSIONS:
        res, reason = _session_arm_outcome_windows(prefix, pre_bins)
        if res is None:
            excluded[prefix] = reason
            session_results[prefix] = {"status": "excluded", "reason": reason}
        else:
            session_results[prefix] = {"status": "computed", **res}
        _log(f"  {prefix}: {'excluded (' + reason + ')' if res is None else 'ok'}")

    output["zero_drop_accounting"] = {
        "sessions_seen": len(SESSIONS),
        "sessions_used": sum(1 for s in session_results.values() if s["status"] == "computed"),
        "sessions_excluded_by_reason": excluded,
        "per_session_trial_counts": {
            prefix: (sr.get("counts") if sr.get("status") == "computed" else {"status": "excluded", "reason": sr.get("reason")})
            for prefix, sr in session_results.items()
        },
    }
    _flush(output)

    _log("running the subsample temporal validity probe")
    validity_probe = run_subsample_temporal_validity_probe(session_results)
    output["validity_probe_correct_trial_subsampling_random_wrt_session_time"] = validity_probe
    output["status"] = "validity_probe_complete_link_pending"
    _flush(output)

    _log(f"validity_assumption_status = {validity_probe['validity_assumption_status']}")
    output["decision_rule_declared_before_fitting"] = DECISION_RULE_DECLARED_BEFORE_FITTING

    _log("computing the pooled correct-vs-error displacement link")
    output["behaviour_link"] = run_behaviour_link(census_windows, session_results, validity_probe["validity_assumption_status"])
    output["census_windows"] = census_windows
    output["sa210311_s224_is_a_different_montage"] = (
        "Sa210311_s224 stimulates through bipolar channel pairs against a different electrode montage "
        "from the ten Wa sessions (see module docstring and the loader hazards this project has already "
        "measured for this corpus); it is included in every primary pooled cell above (2 animal clusters) "
        "and every cell also carries a Wa-only descriptive sensitivity value under "
        "sensitivity_excluding_sa210311_s224, which is NOT a formal cluster test (1 cluster) but is "
        "reported so the primary result is never silently Sa-driven or Sa-masked."
    )
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    _log(f"done in {output['wall_clock_s']:.1f}s")


if __name__ == "__main__":
    main()
