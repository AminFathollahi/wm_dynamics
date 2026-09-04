"""run_preparation_observability_gap.py -- how much of the shared cross-unit component
does each recording preparation actually deliver to this project's persistence estimator?

Every human arm in this project comes out underpowered, and that has been read as a
statement about session count. It is at least as much a statement about amplitude, and
the amplitude is measurable, because one animal arm was already built to be
instrumentally comparable to the human delay arm.

results/state_persistence_lag.json (read-only here; never recomputed, never edited)
carries a mouse anterior lateral motor cortex arm re-run at the human delay window
(2.3 s), the same 0.1 s bins, the same 23 bins, the same fixed-width estimator at width
3 bins, the same lag range, and -- the part that makes it a matched comparison rather
than a cross-corpus glance -- with its spike counts binomially thinned to the human
median firing rate before the estimator runs. The keep probability is read back out of
results/state_persistence.json rather than restated here, so the thinning applied to
the refits below is bit-identical to the thinning already in the delivered rows.

The quantity compared is one number per session, called the shared-component amplitude:

    amplitude = mean over the common reachable lag bins of
                [ observed split-half correlation median at lag L ]
              - [ per-unit permutation null's median correlation at lag L ]

The per-unit permutation null destroys cross-unit coordination while leaving each
unit's own spike-train autocorrelation intact, so the difference is the part of the
split-half correlation that requires units to co-vary across trials.

Three things are computed.

1. The matched preparation contrast. Both arms are restricted to pooled-structure,
   delay-epoch, width-3, 2.3 s rows with a fitted profile and a fitted permutation
   null, the reachable lag range common to both arms is found rather than assumed, and
   the arms are compared unpaired -- as a ratio of medians, as a difference with a
   bootstrap confidence interval, and by the overlap of the two distributions, because
   a ratio computed against a near-zero denominator overstates a difference whose
   distributions in fact overlap. The arms are then matched on unit count and trial
   count by subsampling units and trials from the raw counts and re-running the
   identical estimator, at several pre-declared matched cell sizes, rather than by
   restricting each arm to a band of its own sessions (which discards most of the
   human arm and leaves the two arms' within-band distributions unmatched anyway).

2. The human regional amplitude ladder. The human per-structure rows are a
   within-corpus amplitude ladder at fixed task, fixed electrode technology and fixed
   estimator, so they can ask whether any human recording target reaches the animal
   range without changing anything except which units are pooled. Between-structure
   heterogeneity is tested by permuting structure labels WITHIN session, which is the
   exchangeability the design actually supports: each session contributes one row per
   structure it recorded from plus a pooled row over all of them, so structure rows are
   nested in sessions and an unrestricted label shuffle would treat one recording as
   several independent ones. Each structure is also contrasted against its own
   sessions' pooled row, patient-clustered, with one false-discovery-rate pass across
   structures.

3. What the human amplitude implies for the project's human nulls: the smallest true
   amplitude the human delay arm could have detected at 80% power at the session count
   it has, its power against the amplitudes the animal arms deliver, and the session
   count a preparation delivering the human amplitude would need to reach 80% power.

What this cannot say. The two arms differ in brain region (mouse anterior lateral
motor cortex against predominantly human medial temporal and frontal depth contacts),
in task, and in unit isolation quality between acute high-density probes and chronic
clinical microwires. None of those can be matched with these data, so nothing here
separates species from region, task or electrode, and no sentence anywhere in this
module's output attributes the gap to species or claims that human working memory
lacks a shared population state. What is measured is what a preparation delivers to
this estimator.

Long-running by design and therefore crash-proof by design: every subsampled refit is
written to a checkpoint under results/.checkpoints/ as soon as it is computed, through
a numpy-aware serializer, and a re-run skips any refit already on disk.
"""

from __future__ import annotations

import os

# Capped before numpy is imported: several of these jobs run concurrently on one
# machine, and an uncapped BLAS thread pool per process turns a fast fit into a
# scheduling contest.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
_src_dir = str(REPO_ROOT / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(REPO_ROOT / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import (  # noqa: E402
    alm_data_directory, data_root, iter_dandi_000469, iter_dandi_001187, load_alm_raw_session,
)
from provenance import canonical_json  # noqa: E402
from state_persistence import _d_series, binomial_thin, lag_reachability_note  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh, minimum_detectable_paired_difference, paired_sign_flip_test,
    permutation_pvalue, power_to_detect_effect, stable_seed,
)
from run_state_persistence import (  # noqa: E402
    LAG_N_NULL_REPLICATES, LAG_N_SPLITS, _counts_from_spikes, _lag_run_row, _seed,
)
from run_persistence_patient_clustered_replication import (  # noqa: E402
    _one_sample_patient_stats, _paired_patient_stats, _patient_median,
)

LAG_PATH = REPO_ROOT / "results" / "state_persistence_lag.json"
PERSISTENCE_PATH = REPO_ROOT / "results" / "state_persistence.json"
OUTPUT_PATH = REPO_ROOT / "results" / "preparation_observability_gap.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / ".checkpoints" / "preparation_observability_gap_checkpoint.json"

# ── The estimator cell both arms are read at ─────────────────────────────────
WIDTH_BINS = 3
BIN_MS = 100.0
BIN_WIDTH_S = BIN_MS / 1000.0
WINDOW_S = 2.3
EXPECTED_N_BINS = 23
STRUCTURE = "pooled"
EPOCH = "delay"

HUMAN_ARM_LABEL = "human_intracranial_microwire_delay"
MOUSE_ARM_LABEL = "mouse_anterior_lateral_motor_cortex_rate_matched"
MACAQUE_ARM_LABEL = "macaque_lateral_prefrontal_delay"

SEED = 20260813
ALPHA = 0.05
N_BOOT = 4000
N_PERM = 10000
N_SUBSAMPLE_REPEATS = 3

# ── Pre-declared decision rules ──────────────────────────────────────────────
# Written into the script before any matched-cell number was computed. A branch
# invented after seeing a result is not a result.
#
# MATERIAL_AMPLITUDE_DIFFERENCE is half the human delay arm's own delivered existence
# effect against this same null, in these same units. A between-preparation difference
# smaller than half of what the weaker preparation shows on its own cannot support the
# claim that the preparations deliver different amounts of shared component; an
# interval that reaches past it in BOTH directions cannot exclude one either, and that
# is the inconclusive case.
MATERIAL_AMPLITUDE_DIFFERENCE = 0.025

# Matched cells: (units retained, trials retained). Chosen from the two arms' delivered
# unit and trial counts alone -- the loosest cell either arm can supply through to the
# tightest that still leaves both arms above MIN_SESSIONS_PER_ARM_PER_CELL -- and fixed
# before any amplitude at any cell was computed.
MATCHED_CELLS = ((20, 28), (20, 40), (25, 30), (30, 30), (30, 40), (40, 28))
MIN_SESSIONS_PER_ARM_PER_CELL = 10

# The deciding cell is chosen by a rule, not by its result: among the pre-declared
# cells that keep at least MIN_SESSIONS_PER_ARM_PER_CELL sessions in BOTH arms, the one
# retaining the most units and, at equal units, the most trials -- the most instrument-
# matched cell the data can still support.
DECIDING_CELL_RULE = ("among the pre-declared matched cells that keep at least "
                      f"{MIN_SESSIONS_PER_ARM_PER_CELL} sessions in both arms, the one retaining the most units "
                      "and, at equal units, the most trials")

# A structure enters the regional ladder's tests (as opposed to its descriptive table,
# which lists every structure seen) only with at least this many sessions.
MIN_LADDER_SESSIONS = 4

PREPARATION_BRANCHES = {
    "preparation_gap_survives_instrumental_matching": (
        "At the deciding matched cell -- both arms subsampled to the same unit count and the "
        "same trial count and re-run through the identical estimator -- the arms still differ: "
        "the difference in median amplitude clears a single false-discovery-rate pass across "
        "the matched cells and its 95% confidence interval lies entirely beyond "
        f"{MATERIAL_AMPLITUDE_DIFFERENCE} in amplitude units. Whatever sets the amplitude of the shared "
        "component is not unit count and not trial count."
    ),
    "preparation_gap_explained_by_instrumental_matching": (
        "At the deciding matched cell the difference between the arms does not clear the "
        "false-discovery-rate pass and its 95% confidence interval lies entirely inside "
        f"+/-{MATERIAL_AMPLITUDE_DIFFERENCE} in amplitude units. Matching unit count and trial count removes "
        "the difference, so the difference was an instrument-sampling difference."
    ),
    "inconclusive_below_detection_floor": (
        "The deciding matched cell's confidence interval on the difference reaches past "
        f"+/-{MATERIAL_AMPLITUDE_DIFFERENCE} in amplitude units in both directions, or the cell does not "
        "resolve whether the difference is material. The matched comparison cannot separate a "
        "real preparation difference from no difference at this matched cell size."
    ),
}

REGIONAL_BRANCHES = {
    "human_amplitude_is_regionally_heterogeneous": (
        "Between-structure amplitude heterogeneity exceeds what a within-session shuffle of "
        "structure labels produces, at the 0.05 level. Recording target changes how much shared "
        "component the same preparation delivers."
    ),
    "human_amplitude_is_regionally_uniform": (
        "Between-structure heterogeneity does not exceed the within-session label shuffle, AND "
        "every tested structure's patient-clustered contrast against its own sessions' pooled "
        f"amplitude has a confidence interval inside +/-{MATERIAL_AMPLITUDE_DIFFERENCE} amplitude units, so a "
        "material regional difference is excluded rather than merely unobserved."
    ),
    "inconclusive_below_detection_floor": (
        "Between-structure heterogeneity does not clear the within-session label shuffle, but at "
        "least one tested structure's patient-clustered interval still reaches past "
        f"+/-{MATERIAL_AMPLITUDE_DIFFERENCE} amplitude units in both directions. The human arm cannot resolve "
        "whether recording target matters at this session and patient count."
    ),
}

NOT_MATCHED_BY_THIS_COMPARISON = [
    "Brain region. The animal arm is mouse anterior lateral motor cortex; the human arm is "
    "predominantly medial temporal and frontal depth contacts. No subsampling can match this.",
    "Task. The animal arm is a whisker-based delayed directional response; the human arm is a "
    "visual working-memory maintenance delay. The memoranda are not the same kind of thing.",
    "Unit isolation quality. Acute high-density silicon probes against chronic clinical "
    "microwires implanted for seizure localization. Rate is matched by thinning and unit count "
    "is matched by subsampling, but isolation quality is not measured here and is not matched.",
    "Behavioural state and recording duration. Head-fixed trained animals against clinical "
    "recordings around patient care. Not matched and not measurable from these arms.",
]


# ── Numpy-aware, crash-tolerant serialization ────────────────────────────────

def _write_json(path: Path, payload: dict) -> None:
    """Serialize through the project's numpy-aware canonical encoder and replace
    the target atomically, so a process killed mid-write leaves the previous
    complete file rather than a truncated one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(canonical_json(payload))
    temporary.replace(path)


def _load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        return json.loads(CHECKPOINT_PATH.read_text())
    except json.JSONDecodeError:
        return {}


# ── Amplitude, from delivered rows and from refits alike ─────────────────────

def _int_keyed(lags: dict) -> dict[int, dict]:
    return {int(lag): value for lag, value in lags.items()}


def row_amplitude(row: dict, lag_bins: list[int]) -> float | None:
    """The session's shared-component amplitude over a fixed lag range: the mean of
    the delivered existence statistic d(L) = observed r_median(L) minus the per-unit
    permutation null's r_null_median(L). Computed through the same helper the
    delivered contrasts use so the two can never drift apart. None if the row does
    not reach every requested lag, which is an exclusion and never a silent
    shortening of the mean."""
    if row["profile"].get("status") != "fitted" or row.get("null_permutation") is None:
        return None
    series, = _d_series([_int_keyed(row["profile"]["lags"])],
                        [_int_keyed(row["null_permutation"]["lags"])], "r_null_median")
    if not set(lag_bins) <= set(series):
        return None
    return float(np.mean([series[lag] for lag in lag_bins]))


def reachable_lags(row: dict) -> set[int]:
    if row["profile"].get("status") != "fitted" or row.get("null_permutation") is None:
        return set()
    return {int(lag) for lag in row["profile"]["lags"]} & {int(lag) for lag in row["null_permutation"]["lags"]}


def select_arm_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """The arm's rows at the estimator cell both arms share, and an itemised record
    of every row of that arm that was dropped and why, so that seen = kept + dropped
    reconciles for a reader who never sees this code."""
    kept, dropped = [], []
    for row in rows:
        if row["epoch"] != EPOCH or row["structure"] != STRUCTURE:
            continue
        if row["width_bins"] != WIDTH_BINS:
            continue
        reason = None
        if row["bin_ms"] != BIN_MS:
            reason = "bin_width_is_not_the_shared_0_1_s_grid"
        elif row["window_s"] != WINDOW_S:
            reason = "window_length_is_not_the_shared_2_3_s_window"
        elif row["profile"].get("status") != "fitted":
            reason = f"profile_not_fitted__{row['profile'].get('status')}"
        elif row.get("null_permutation") is None:
            reason = "per_unit_permutation_null_absent"
        if reason is None:
            kept.append(row)
        else:
            dropped.append({"dataset": row["dataset"], "session": row["session"],
                            "window_s": row["window_s"], "reason": reason})
    return kept, dropped


# ── Unpaired arm contrast ────────────────────────────────────────────────────

def _median_difference(pair: np.ndarray) -> float:
    return float(np.median(pair[:, 0]) - np.median(pair[:, 1]))


def probability_of_superiority(higher: np.ndarray, lower: np.ndarray) -> float:
    """The probability that a randomly drawn session of the first arm exceeds a
    randomly drawn session of the second, ties counted as half -- the overlap
    statistic a ratio of medians cannot express."""
    comparisons = higher[:, None] - lower[None, :]
    return float(((comparisons > 0).sum() + 0.5 * (comparisons == 0).sum()) / comparisons.size)


def unpaired_arm_contrast(reference: np.ndarray, comparison: np.ndarray, seed_parts: tuple) -> dict:
    """Unpaired contrast of two arms' per-session amplitudes, reported four ways
    because no one of them is sufficient: a ratio of medians (which inflates without
    bound as the denominator approaches zero), a difference of medians with a
    bootstrap interval (which does not), a rank test, and the overlap of the two
    distributions."""
    if len(reference) < 2 or len(comparison) < 2:
        return {"status": "not_computable", "n_reference": len(reference), "n_comparison": len(comparison)}
    rng = np.random.default_rng((stable_seed("|".join(str(p) for p in seed_parts)) ^ SEED) & 0xFFFFFFFF)
    # Bootstrapped arm by arm: the arms are unpaired, so resampling them jointly would
    # tie their resample sizes to each other.
    observed = float(np.median(comparison) - np.median(reference))
    boot = np.array([
        np.median(comparison[rng.integers(0, len(comparison), len(comparison))])
        - np.median(reference[rng.integers(0, len(reference), len(reference))])
        for _ in range(N_BOOT)
    ])
    ci_lower, ci_upper = (float(v) for v in np.nanpercentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
    mann_whitney = stats.mannwhitneyu(comparison, reference, alternative="two-sided")
    reference_median = float(np.median(reference))
    return {
        "status": "computed",
        "n_reference": int(len(reference)),
        "n_comparison": int(len(comparison)),
        "reference_median_amplitude": reference_median,
        "comparison_median_amplitude": float(np.median(comparison)),
        "reference_mean_amplitude": float(np.mean(reference)),
        "comparison_mean_amplitude": float(np.mean(comparison)),
        "reference_amplitude_iqr": [float(np.percentile(reference, 25)), float(np.percentile(reference, 75))],
        "comparison_amplitude_iqr": [float(np.percentile(comparison, 25)), float(np.percentile(comparison, 75))],
        "median_difference_comparison_minus_reference": observed,
        "median_difference_ci_95": [ci_lower, ci_upper],
        "ratio_of_medians": (float(np.median(comparison) / reference_median)
                             if reference_median != 0 else None),
        "ratio_is_unstable_near_zero_denominator": bool(abs(reference_median) < MATERIAL_AMPLITUDE_DIFFERENCE),
        "mann_whitney_u_p_value": float(mann_whitney.pvalue),
        "overlap": {
            "reference_max": float(np.max(reference)),
            "comparison_min": float(np.min(comparison)),
            "distributions_are_disjoint": bool(np.max(reference) < np.min(comparison)),
            "n_reference_above_comparison_median": int((reference > np.median(comparison)).sum()),
            "n_comparison_below_reference_median": int((comparison < reference_median).sum()),
            "probability_comparison_exceeds_reference": probability_of_superiority(comparison, reference),
        },
    }


# ── Raw counts for the matched-cell refits ───────────────────────────────────

def human_session_counts(root: Path) -> tuple[dict[str, dict], list[dict]]:
    """Delay-epoch spike counts at the shared estimator grid for every human
    pooled-structure session of the two datasets whose delay window is the shared
    2.3 s, keyed by dataset-qualified session identifier -- dandi_000469 and
    dandi_001187 reuse the same BIDS subject/session numbering (e.g. both have a
    "sub-1_ses-2..." file), so keying by the bare session stem silently drops
    whichever of the two loads second. The third human dataset's delay window is
    3.0 s and its rows never enter this comparison; it is recorded as an exclusion
    rather than dropped silently."""
    sessions, notes = {}, []
    for iterator, dataset in ((iter_dandi_000469, "dandi_000469"), (iter_dandi_001187, "dandi_001187")):
        for meta in iterator(root):
            if meta["structure"] != STRUCTURE:
                continue
            window_s = meta["epoch_windows"][EPOCH]
            if window_s != WINDOW_S:
                notes.append({"dataset": dataset, "session": meta["session"],
                              "reason": "native_delay_window_is_not_the_shared_2_3_s_window",
                              "window_s": window_s})
                continue
            counts = _counts_from_spikes(meta["spike_lists"], meta["epoch_onsets"][EPOCH], window_s, BIN_MS)
            sessions[f"{dataset}:{meta['session']}"] = {
                "arm": HUMAN_ARM_LABEL, "dataset": dataset, "patient": meta["patient"],
                "session": meta["session"], "counts": counts,
            }
    return sessions, notes


def mouse_session_counts(root: Path, keep_probability: float) -> tuple[dict[str, dict], list[dict]]:
    """The mouse arm's counts at the same grid, thinned with the same keep
    probability and the same seed the delivered rows used, so the counts refit here
    are the counts already fitted there."""
    directory = alm_data_directory(root)
    sessions, notes = {}, []
    if not directory.is_dir():
        return sessions, [{"reason": "mouse_data_directory_not_staged", "path": str(directory)}]
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=BIN_MS, window_s=WINDOW_S, require_both_arms=False)
        if session is None:
            notes.append({"session": path.stem, "reason": "session_not_loadable_at_the_shared_window"})
            continue
        thin_rng = np.random.default_rng(_seed("alm_lag_thin", path.stem))
        counts = binomial_thin(session["control_counts"], keep_probability, thin_rng)
        sessions[path.stem] = {
            "arm": MOUSE_ARM_LABEL, "dataset": "inagaki_alm5", "patient": path.stem,
            "session": path.stem, "counts": counts,
        }
    return sessions, notes


def counts_match_delivered_row(counts: np.ndarray, row: dict) -> bool:
    return (counts.shape[0] == row["n_trials"] and counts.shape[1] == row["n_units"]
            and counts.shape[2] == row["n_bins"])


# ── Matched-cell refits ──────────────────────────────────────────────────────

def matched_cell_amplitude(entry: dict, cell: tuple[int, int], lag_bins: list[int],
                            checkpoint: dict, on_flush) -> dict:
    """One session's amplitude after being subsampled to a matched cell, averaged
    over independent subsample draws so that a single unlucky draw does not decide a
    session. Every draw is checkpointed the moment it is computed: this loop is the
    expensive part of the module and must survive being killed."""
    n_units, n_trials = cell
    counts = entry["counts"]
    if counts.shape[1] < n_units or counts.shape[0] < n_trials:
        return {"status": "session_smaller_than_cell",
                "n_units_available": int(counts.shape[1]), "n_trials_available": int(counts.shape[0])}
    draws = []
    for repeat in range(N_SUBSAMPLE_REPEATS):
        # dataset, not just arm, must qualify the key: dandi_000469 and dandi_001187
        # reuse the same session stems (see human_session_counts), and without the
        # dataset the two sessions' cached fits would overwrite each other.
        key = f"{entry['arm']}|{entry['dataset']}|{entry['session']}|u{n_units}t{n_trials}|r{repeat}"
        cached = checkpoint.get(key)
        if cached is None:
            rng = np.random.default_rng((stable_seed(key) ^ SEED) & 0xFFFFFFFF)
            unit_index = rng.choice(counts.shape[1], size=n_units, replace=False)
            trial_index = rng.choice(counts.shape[0], size=n_trials, replace=False)
            subsampled = counts[np.ix_(trial_index, unit_index, np.arange(counts.shape[2]))]
            run = _lag_run_row(subsampled, WIDTH_BINS, int(stable_seed(key + "|fit")) % (2 ** 31))
            cached = {"amplitude": row_amplitude(run, lag_bins),
                      "profile_status": run["profile"].get("status")}
            checkpoint[key] = cached
            on_flush()
        if cached["amplitude"] is not None:
            draws.append(float(cached["amplitude"]))
    if not draws:
        return {"status": "no_draw_fitted", "n_draws_attempted": N_SUBSAMPLE_REPEATS}
    return {"status": "computed", "amplitude": float(np.mean(draws)), "n_draws": len(draws),
            "draw_spread": float(np.std(draws, ddof=1)) if len(draws) > 1 else 0.0}


def run_matched_cells(entries: dict[str, dict], lag_bins: list[int], checkpoint: dict, on_flush) -> dict:
    """Every pre-declared matched cell, every session, both arms. Reports per cell the
    sessions each arm could supply and the sessions it could not, so the shrinking n
    down the cell ladder is visible rather than implied."""
    per_cell = {}
    for cell in MATCHED_CELLS:
        label = f"units_{cell[0]}_trials_{cell[1]}"
        by_arm: dict[str, list[dict]] = {}
        excluded: list[dict] = []
        for entry in entries.values():
            result = matched_cell_amplitude(entry, cell, lag_bins, checkpoint, on_flush)
            if result["status"] != "computed":
                excluded.append({"arm": entry["arm"], "dataset": entry["dataset"],
                                  "session": entry["session"], **result})
                continue
            by_arm.setdefault(entry["arm"], []).append({
                "session": entry["session"], "patient": entry["patient"], "dataset": entry["dataset"],
                "amplitude": result["amplitude"], "draw_spread": result["draw_spread"],
            })
        human = np.array([r["amplitude"] for r in by_arm.get(HUMAN_ARM_LABEL, [])], dtype=float)
        mouse = np.array([r["amplitude"] for r in by_arm.get(MOUSE_ARM_LABEL, [])], dtype=float)
        contrast = unpaired_arm_contrast(human, mouse, ("matched_cell", label))
        per_cell[label] = {
            "units_retained": cell[0], "trials_retained": cell[1],
            "n_human_sessions": int(len(human)), "n_mouse_sessions": int(len(mouse)),
            "meets_minimum_sessions_per_arm": bool(len(human) >= MIN_SESSIONS_PER_ARM_PER_CELL
                                                    and len(mouse) >= MIN_SESSIONS_PER_ARM_PER_CELL),
            "contrast": contrast,
            "n_sessions_excluded": len(excluded),
            "excluded_sessions": sorted(excluded, key=lambda item: (item["arm"], item["session"])),
        }
        print(f"  matched cell {label}: human n={len(human)}, mouse n={len(mouse)}", file=sys.stderr, flush=True)
    return per_cell


def summarise_matched_cells(per_cell: dict) -> dict:
    """One false-discovery-rate pass across the matched cells, then the deciding
    cell picked by the pre-declared rule and the branch it fires -- with the effect
    size and the interval that produced it in the same object as the verdict."""
    eligible = {label: cell for label, cell in per_cell.items()
                if cell["meets_minimum_sessions_per_arm"] and cell["contrast"]["status"] == "computed"}
    if not eligible:
        return {"status": "no_cell_meets_minimum_sessions_per_arm",
                "minimum_sessions_per_arm": MIN_SESSIONS_PER_ARM_PER_CELL}
    labels = sorted(eligible)
    p_values = np.array([eligible[label]["contrast"]["mann_whitney_u_p_value"] for label in labels])
    fdr = fdr_bh(p_values, alpha=ALPHA)
    corrected = {
        label: {"p_value": float(p), "q_value": float(q), "clears_fdr": bool(reject)}
        for label, p, q, reject in zip(labels, p_values, fdr["q_values"], fdr["reject"])
    }
    deciding_label = max(eligible, key=lambda label: (eligible[label]["units_retained"],
                                                       eligible[label]["trials_retained"]))
    deciding = eligible[deciding_label]
    contrast = deciding["contrast"]
    ci_lower, ci_upper = contrast["median_difference_ci_95"]
    clears = corrected[deciding_label]["clears_fdr"]
    if clears and ci_lower > MATERIAL_AMPLITUDE_DIFFERENCE:
        branch = "preparation_gap_survives_instrumental_matching"
    elif (not clears) and abs(ci_lower) < MATERIAL_AMPLITUDE_DIFFERENCE and abs(ci_upper) < MATERIAL_AMPLITUDE_DIFFERENCE:
        branch = "preparation_gap_explained_by_instrumental_matching"
    else:
        branch = "inconclusive_below_detection_floor"
    signs = {label: int(np.sign(eligible[label]["contrast"]["median_difference_comparison_minus_reference"]))
             for label in labels}
    return {
        "status": "computed",
        "cells_entering_correction": labels,
        "false_discovery_rate_across_cells": corrected,
        "n_cells_clearing_fdr": int(fdr["n_reject"]),
        "deciding_cell": deciding_label,
        "deciding_cell_rule": DECIDING_CELL_RULE,
        "all_cells_agree_in_sign": len(set(signs.values())) == 1,
        "sign_by_cell": signs,
        "branch": branch,
        "branch_meaning": PREPARATION_BRANCHES[branch],
        "effect_size_at_deciding_cell": {
            "n_human_sessions": deciding["n_human_sessions"],
            "n_mouse_sessions": deciding["n_mouse_sessions"],
            "human_median_amplitude": contrast["reference_median_amplitude"],
            "mouse_median_amplitude": contrast["comparison_median_amplitude"],
            "median_difference_mouse_minus_human": contrast["median_difference_comparison_minus_reference"],
            "median_difference_ci_95": contrast["median_difference_ci_95"],
            "ratio_of_medians": contrast["ratio_of_medians"],
            "mann_whitney_u_p_value": contrast["mann_whitney_u_p_value"],
            "false_discovery_rate_q_value": corrected[deciding_label]["q_value"],
            "probability_a_mouse_session_exceeds_a_human_session":
                contrast["overlap"]["probability_comparison_exceeds_reference"],
            "material_difference_threshold": MATERIAL_AMPLITUDE_DIFFERENCE,
        },
    }


# ── The human regional amplitude ladder ──────────────────────────────────────

def ladder_rows(lag_rows: list[dict], lag_bins: list[int]) -> tuple[list[dict], list[dict]]:
    """Every human delay row at the shared estimator width that reaches every lag in
    the ladder's range, pooled and per-structure alike, carrying the amplitude and the
    identifiers the nesting corrections need. Rows are kept across BOTH human delay
    window lengths here (unlike the preparation contrast, which needs the animal arm's
    window): this ladder is a within-corpus comparison and the lag range is chosen so
    every window reaches it."""
    kept, dropped = [], []
    for row in lag_rows:
        if row["epoch"] != EPOCH or row["width_bins"] != WIDTH_BINS or row["bin_ms"] != BIN_MS:
            continue
        amplitude = row_amplitude(row, lag_bins)
        if amplitude is None:
            dropped.append({"dataset": row["dataset"], "session": row["session"],
                            "structure": row["structure"],
                            "reason": ("profile_not_fitted_or_null_absent"
                                       if row["profile"].get("status") != "fitted"
                                       else "row_does_not_reach_every_ladder_lag")})
            continue
        kept.append({
            "dataset": row["dataset"], "patient": f"{row['dataset']}:{row['patient']}",
            "session": f"{row['dataset']}:{row['session']}", "structure": row["structure"],
            "window_s": row["window_s"], "n_units": row["n_units"], "n_trials": row["n_trials"],
            "amplitude": amplitude,
        })
    return kept, dropped


def structure_descriptives(rows: list[dict]) -> dict:
    by_structure: dict[str, list[dict]] = {}
    for row in rows:
        by_structure.setdefault(row["structure"], []).append(row)
    out = {}
    for structure, group in by_structure.items():
        amplitudes = np.array([r["amplitude"] for r in group], dtype=float)
        out[structure] = {
            "n_sessions": len(group),
            "n_patients": len({r["patient"] for r in group}),
            "datasets": sorted({r["dataset"] for r in group}),
            "median_units": float(np.median([r["n_units"] for r in group])),
            "median_trials": float(np.median([r["n_trials"] for r in group])),
            "mean_amplitude": float(np.mean(amplitudes)),
            "median_amplitude": float(np.median(amplitudes)),
            "amplitude_iqr": [float(np.percentile(amplitudes, 25)), float(np.percentile(amplitudes, 75))],
            "enters_tests": len(group) >= MIN_LADDER_SESSIONS,
        }
    return dict(sorted(out.items(), key=lambda item: -item[1]["mean_amplitude"]))


def within_session_heterogeneity(rows: list[dict], seed_parts: tuple) -> dict:
    """Is between-structure amplitude spread larger than chance? The null shuffles
    structure labels WITHIN each session, which is the only shuffle the design
    supports: one session contributes one row per structure it recorded, so those
    rows are nested and a free shuffle would treat one recording as several.

    Sessions that contributed only one structure row are invariant under this shuffle
    and so contribute no evidence; they are counted and reported rather than being
    allowed to inflate a session count the test cannot actually use."""
    tested = [r for r in rows if r["structure"] != STRUCTURE]
    by_structure: dict[str, list[float]] = {}
    for row in tested:
        by_structure.setdefault(row["structure"], []).append(row["amplitude"])
    eligible_structures = {s for s, v in by_structure.items() if len(v) >= MIN_LADDER_SESSIONS}
    tested = [r for r in tested if r["structure"] in eligible_structures]
    if len(eligible_structures) < 2:
        return {"status": "not_computable", "reason": "fewer than two structures reach the minimum session count",
                "minimum_sessions": MIN_LADDER_SESSIONS}

    positions_by_session: dict[str, list[int]] = {}
    for position, row in enumerate(tested):
        positions_by_session.setdefault(row["session"], []).append(position)
    shuffleable = [positions for positions in positions_by_session.values()
                   if len({tested[p]["structure"] for p in positions}) >= 2]

    amplitudes = np.array([row["amplitude"] for row in tested], dtype=float)

    def statistic(labels: np.ndarray) -> float:
        groups = [amplitudes[labels == label] for label in np.unique(labels)]
        present = [g for g in groups if len(g) >= 2]
        if len(present) < 2:
            return 0.0
        return float(stats.kruskal(*present).statistic)

    observed_labels = np.array([row["structure"] for row in tested])
    observed = statistic(observed_labels)
    rng = np.random.default_rng((stable_seed("|".join(str(p) for p in seed_parts)) ^ SEED) & 0xFFFFFFFF)
    null = np.empty(N_PERM)
    for draw in range(N_PERM):
        labels = observed_labels.copy()
        for positions in shuffleable:
            labels[positions] = rng.permutation(labels[positions])
        null[draw] = statistic(labels)
    p_value = permutation_pvalue(null >= observed)

    naive = stats.kruskal(*[by_structure[s] for s in sorted(eligible_structures)])
    return {
        "status": "computed",
        "structures_tested": sorted(eligible_structures),
        "n_rows": len(tested),
        "n_sessions": len(positions_by_session),
        "n_sessions_contributing_to_the_null": len(shuffleable),
        "n_patients": len({r["patient"] for r in tested}),
        "kruskal_wallis_h": observed,
        "within_session_label_permutation_p_value": float(p_value),
        "n_permutations": N_PERM,
        "null_h_median": float(np.median(null)),
        "null_h_p95": float(np.percentile(null, 95)),
        "unclustered_kruskal_wallis_p_value_sensitivity_check": float(naive.pvalue),
        "reachability": ("the Kruskal-Wallis statistic is unbounded above and its within-session "
                         "permutation null is built from the same rows, so a spread larger than chance "
                         "is reachable whenever at least two structures are recorded in the same session; "
                         f"{len(shuffleable)} sessions meet that condition here"),
    }


def structure_versus_own_pooled(rows: list[dict], seed_parts: tuple) -> dict:
    """For each structure, the within-session contrast against the SAME session's
    pooled row -- the comparison that asks whether restricting a human population to
    one recording target raises the amplitude the estimator sees, holding the session
    fixed. Reported session-level and patient-clustered, with one false-discovery-rate
    pass across structures."""
    pooled_by_session = {r["session"]: r for r in rows if r["structure"] == STRUCTURE}
    by_structure: dict[str, list[dict]] = {}
    for row in rows:
        if row["structure"] == STRUCTURE:
            continue
        if row["session"] not in pooled_by_session:
            continue
        by_structure.setdefault(row["structure"], []).append(row)

    per_structure, p_values, labels = {}, [], []
    for structure in sorted(by_structure):
        group = by_structure[structure]
        if len(group) < MIN_LADDER_SESSIONS:
            per_structure[structure] = {"status": "below_minimum_sessions", "n_sessions": len(group),
                                        "minimum_sessions": MIN_LADDER_SESSIONS}
            continue
        structure_values = np.array([r["amplitude"] for r in group], dtype=float)
        pooled_values = np.array([pooled_by_session[r["session"]]["amplitude"] for r in group], dtype=float)
        session_level = paired_sign_flip_test(
            structure_values, pooled_values, alternative="two-sided",
            rng=np.random.default_rng((stable_seed(f"{structure}|session") ^ SEED) & 0xFFFFFFFF))
        structure_by_patient = _patient_median({(r["patient"], r["session"]): r["amplitude"] for r in group})
        pooled_by_patient = _patient_median(
            {(r["patient"], r["session"]): pooled_by_session[r["session"]]["amplitude"] for r in group})
        patient_level = _paired_patient_stats(structure_by_patient, pooled_by_patient,
                                               seed_parts + (structure,))
        absolute = _one_sample_patient_stats(structure_by_patient)
        per_structure[structure] = {
            "status": "computed",
            "n_sessions": len(group),
            "n_patients": len(structure_by_patient),
            "median_structure_amplitude": float(np.median(structure_values)),
            "median_own_session_pooled_amplitude": float(np.median(pooled_values)),
            "median_difference_structure_minus_pooled": float(np.median(structure_values - pooled_values)),
            "n_sessions_structure_above_own_pooled": int((structure_values > pooled_values).sum()),
            "session_level": {
                "mean_difference": float(np.mean(structure_values - pooled_values)),
                "p_value": session_level["p_value"],
                "ci_95": [session_level["ci_lower"], session_level["ci_upper"]],
            },
            "patient_clustered": patient_level,
            "patient_clustered_amplitude_against_zero": absolute,
        }
        if patient_level.get("status") == "computed":
            labels.append(structure)
            p_values.append(patient_level["p_value"])

    correction = {}
    if p_values:
        fdr = fdr_bh(np.array(p_values), alpha=ALPHA)
        correction = {
            label: {"p_value": float(p), "q_value": float(q), "clears_fdr": bool(reject)}
            for label, p, q, reject in zip(labels, p_values, fdr["q_values"], fdr["reject"])
        }
        for label in labels:
            per_structure[label]["patient_clustered_false_discovery_rate"] = correction[label]
    return {"per_structure": per_structure,
            "false_discovery_rate_across_structures": correction,
            "n_structures_corrected": len(labels),
            "n_structures_clearing_fdr": sum(1 for v in correction.values() if v["clears_fdr"])}


def regional_verdict(heterogeneity: dict, contrasts: dict) -> dict:
    """The regional branch, with the effect sizes that produced it beside it."""
    if heterogeneity.get("status") != "computed":
        return {"branch": "inconclusive_below_detection_floor",
                "branch_meaning": REGIONAL_BRANCHES["inconclusive_below_detection_floor"],
                "reason": heterogeneity.get("reason", "heterogeneity test not computable")}
    p_value = heterogeneity["within_session_label_permutation_p_value"]
    intervals = []
    for structure, result in contrasts["per_structure"].items():
        patient = result.get("patient_clustered", {})
        if patient.get("status") != "computed":
            continue
        intervals.append((structure, patient["ci_lower_mean_difference"], patient["ci_upper_mean_difference"]))
    reaches_material = [
        structure for structure, low, high in intervals
        if low is not None and high is not None
        and (abs(low) >= MATERIAL_AMPLITUDE_DIFFERENCE or abs(high) >= MATERIAL_AMPLITUDE_DIFFERENCE)
    ]
    if p_value < ALPHA:
        branch = "human_amplitude_is_regionally_heterogeneous"
    elif not reaches_material and intervals:
        branch = "human_amplitude_is_regionally_uniform"
    else:
        branch = "inconclusive_below_detection_floor"
    ranked = sorted(
        ((structure, result) for structure, result in contrasts["per_structure"].items()
         if result.get("status") == "computed"),
        key=lambda item: -item[1]["median_difference_structure_minus_pooled"])
    return {
        "branch": branch,
        "branch_meaning": REGIONAL_BRANCHES[branch],
        "within_session_label_permutation_p_value": p_value,
        "kruskal_wallis_h": heterogeneity["kruskal_wallis_h"],
        "n_structures_tested": len(heterogeneity["structures_tested"]),
        "n_sessions_contributing_to_the_null": heterogeneity["n_sessions_contributing_to_the_null"],
        "structures_whose_interval_reaches_a_material_difference": reaches_material,
        "highest_structure": ranked[0][0] if ranked else None,
        "highest_structure_effect": ({
            "n_sessions": ranked[0][1]["n_sessions"],
            "n_patients": ranked[0][1]["n_patients"],
            "median_amplitude": ranked[0][1]["median_structure_amplitude"],
            "median_difference_against_own_session_pooled": ranked[0][1]["median_difference_structure_minus_pooled"],
            "patient_clustered_p_value": ranked[0][1]["patient_clustered"].get("p_value"),
            "patient_clustered_q_value": ranked[0][1].get("patient_clustered_false_discovery_rate", {}).get("q_value"),
            "patient_clustered_ci_95": [ranked[0][1]["patient_clustered"].get("ci_lower_mean_difference"),
                                         ranked[0][1]["patient_clustered"].get("ci_upper_mean_difference")],
        } if ranked else None),
        "material_difference_threshold": MATERIAL_AMPLITUDE_DIFFERENCE,
    }


# ── What the human amplitude implies for every human null in this project ────

def sessions_needed(effect: float, sd: float, power: float = 0.80, alpha: float = ALPHA) -> int | None:
    """Sessions a one-sample two-sided test needs to detect a true amplitude of
    ``effect`` given between-session spread ``sd``, by the normal approximation the
    project's existing minimum-detectable-difference helper uses, so the two numbers
    are on the same footing."""
    if effect == 0 or sd <= 0 or not math.isfinite(effect) or not math.isfinite(sd):
        return None
    z = float(stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power))
    return int(math.ceil((z * sd / abs(effect)) ** 2))


def human_arm_detectability(human: np.ndarray, mouse: np.ndarray, macaque: np.ndarray | None) -> dict:
    """The smallest amplitude the human arm could have detected at its own session
    count, its power against the amplitudes the animal arms deliver, and -- the
    statement this project has needed -- how many sessions a preparation delivering the
    human amplitude would need before it could clear the same test the animal arms
    clear."""
    sd = float(np.std(human, ddof=1))
    mean = float(np.mean(human))
    mdd = minimum_detectable_paired_difference(human, alpha=ALPHA, power=0.80)
    targets = {"human_arm_own_mean_amplitude": mean,
               "mouse_arm_median_amplitude": float(np.median(mouse))}
    if macaque is not None and len(macaque) >= 2:
        targets["macaque_arm_median_amplitude"] = float(np.median(macaque))
    against = {}
    for name, effect in targets.items():
        against[name] = {
            "effect": effect,
            "power_of_the_human_arm_at_its_own_session_count": power_to_detect_effect(effect, human, alpha=ALPHA),
            "human_sessions_needed_for_80_percent_power": sessions_needed(effect, sd),
        }
    return {
        "n_human_sessions": int(len(human)),
        "human_mean_amplitude": mean,
        "human_median_amplitude": float(np.median(human)),
        "human_between_session_sd": sd,
        "minimum_detectable_amplitude_at_80_percent_power": mdd,
        "power_and_required_sessions_against": against,
        "amplitude_ratios": {
            "mouse_over_human_medians": (float(np.median(mouse) / np.median(human))
                                          if np.median(human) != 0 else None),
            "macaque_over_human_medians": (float(np.median(macaque) / np.median(human))
                                            if macaque is not None and len(macaque) and np.median(human) != 0
                                            else None),
        },
        "reading": ("the minimum detectable amplitude is the smallest true shared-component amplitude a "
                    "one-sample test at this session count and this between-session spread could have "
                    "found at 80% power; a human null on any quantity derived from this state is "
                    "uninformative about effects below it"),
    }


# ── Assembly ─────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.time()
    lag_artifact = json.loads(LAG_PATH.read_text())
    persistence = json.loads(PERSISTENCE_PATH.read_text())
    keep_probability = float(persistence["matched_sensitivity_alm"]["rate_matched_keep_probability"])

    human_rows, human_dropped = select_arm_rows(lag_artifact["human_lag_rows"])
    mouse_rows, mouse_dropped = select_arm_rows(lag_artifact["alm_lag_rows"])
    if not human_rows or not mouse_rows:
        raise SystemExit("neither arm can be selected at the shared estimator cell")

    human_lags = set.intersection(*(reachable_lags(row) for row in human_rows))
    mouse_lags = set.intersection(*(reachable_lags(row) for row in mouse_rows))
    common_lags = sorted(human_lags & mouse_lags)
    if not common_lags:
        raise SystemExit("the two arms share no reachable lag")

    matching = {
        "shared_estimator_cell": {
            "width_bins": WIDTH_BINS, "bin_width_s": BIN_WIDTH_S, "window_s": WINDOW_S,
            "structure": STRUCTURE, "epoch": EPOCH,
            "n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES,
            "null": "per-unit permutation null, which destroys cross-unit coordination and leaves each "
                    "unit's own spike-train autocorrelation intact",
            "matched_cell_refits_use_the_same_settings": True,
        },
        "human_arm": {
            "label": HUMAN_ARM_LABEL, "n_sessions": len(human_rows),
            "n_bins": sorted({row["n_bins"] for row in human_rows}),
            "n_patients": len({(row["dataset"], row["patient"]) for row in human_rows}),
            "datasets": sorted({row["dataset"] for row in human_rows}),
            "median_units": float(np.median([row["n_units"] for row in human_rows])),
            "median_trials": float(np.median([row["n_trials"] for row in human_rows])),
            "unit_count_range": [int(min(r["n_units"] for r in human_rows)), int(max(r["n_units"] for r in human_rows))],
            "trial_count_range": [int(min(r["n_trials"] for r in human_rows)), int(max(r["n_trials"] for r in human_rows))],
        },
        "mouse_arm": {
            "label": MOUSE_ARM_LABEL, "n_sessions": len(mouse_rows),
            "n_bins": sorted({row["n_bins"] for row in mouse_rows}),
            "n_patients": len({row["patient"] for row in mouse_rows}),
            "datasets": sorted({row["dataset"] for row in mouse_rows}),
            "median_units": float(np.median([row["n_units"] for row in mouse_rows])),
            "median_trials": float(np.median([row["n_trials"] for row in mouse_rows])),
            "unit_count_range": [int(min(r["n_units"] for r in mouse_rows)), int(max(r["n_units"] for r in mouse_rows))],
            "trial_count_range": [int(min(r["n_trials"] for r in mouse_rows)), int(max(r["n_trials"] for r in mouse_rows))],
            "firing_rate_thinning_keep_probability": keep_probability,
        },
        "bin_counts_agree": (sorted({row["n_bins"] for row in human_rows})
                             == sorted({row["n_bins"] for row in mouse_rows}) == [EXPECTED_N_BINS]),
        "common_reachable_lag_range_bins": [common_lags[0], common_lags[-1]],
        "common_reachable_lag_range_s": [common_lags[0] * BIN_WIDTH_S, common_lags[-1] * BIN_WIDTH_S],
        "lag_range_is_contiguous": common_lags == list(range(common_lags[0], common_lags[-1] + 1)),
    }
    if not matching["bin_counts_agree"]:
        raise SystemExit("the two arms no longer share a bin count -- the comparison is not matched")

    human_amplitudes = np.array([row_amplitude(row, common_lags) for row in human_rows], dtype=float)
    mouse_amplitudes = np.array([row_amplitude(row, common_lags) for row in mouse_rows], dtype=float)
    unmatched = unpaired_arm_contrast(human_amplitudes, mouse_amplitudes, ("unmatched",))

    macaque_rows = [row for row in lag_artifact.get("panichello_lag_arm", {}).get("rows", [])
                    if row["width_bins"] == WIDTH_BINS and row["profile"].get("status") == "fitted"
                    and row.get("null_permutation") is not None]
    macaque_lags = sorted(set.intersection(*(reachable_lags(row) for row in macaque_rows))) if macaque_rows else []
    macaque_amplitudes = (np.array([row_amplitude(row, macaque_lags) for row in macaque_rows], dtype=float)
                          if macaque_lags else np.array([]))
    macaque_context = {
        "status": "reported_as_context_only",
        "why": ("this arm is not window-matched or split-count-matched to the other two -- its native "
                "delay window is shorter, so it reaches fewer lags, and it is fitted at a different "
                "number of splits -- so its amplitude is a scale reference for the human arm's power "
                "calculation and never a term in the matched contrast"),
        "n_sessions": len(macaque_rows),
        "window_s": sorted({row["window_s"] for row in macaque_rows}) if macaque_rows else [],
        "n_bins": sorted({row["n_bins"] for row in macaque_rows}) if macaque_rows else [],
        "lag_range_bins": [macaque_lags[0], macaque_lags[-1]] if macaque_lags else None,
        "median_amplitude": float(np.median(macaque_amplitudes)) if len(macaque_amplitudes) else None,
    }

    print(f"arms selected: human n={len(human_rows)}, mouse n={len(mouse_rows)}, "
          f"lags {common_lags[0]}-{common_lags[-1]}", file=sys.stderr, flush=True)

    # ── matched-cell refits, checkpointed ──
    root = data_root()
    checkpoint = _load_checkpoint()
    n_at_start = len(checkpoint)
    last_flush = [time.time()]

    def flush() -> None:
        # Flushed on a short timer rather than on every draw: the checkpoint is the
        # only thing standing between a killed process and losing hours of refits,
        # but rewriting it per draw would dominate the draw's own cost.
        if time.time() - last_flush[0] > 20.0:
            _write_json(CHECKPOINT_PATH, checkpoint)
            last_flush[0] = time.time()

    human_counts, human_load_notes = human_session_counts(root)
    mouse_counts, mouse_load_notes = mouse_session_counts(root, keep_probability)
    entries = {**{f"human|{k}": v for k, v in human_counts.items()},
               **{f"mouse|{k}": v for k, v in mouse_counts.items()}}

    # Keyed by (dataset, session), not session alone: dandi_000469 and dandi_001187
    # reuse the same session stems, so a bare-session key would silently pair a
    # reloaded session's counts against the OTHER dataset's delivered row.
    delivered_by_session = {(row["dataset"], row["session"]): row for row in human_rows + mouse_rows}
    reload_check = {"n_sessions_reloaded": len(entries), "mismatched_sessions": [], "unmatched_to_a_delivered_row": []}
    for key, entry in list(entries.items()):
        delivered = delivered_by_session.get((entry["dataset"], entry["session"]))
        if delivered is None:
            reload_check["unmatched_to_a_delivered_row"].append(
                {"dataset": entry["dataset"], "session": entry["session"]})
            entries.pop(key)
        elif not counts_match_delivered_row(entry["counts"], delivered):
            reload_check["mismatched_sessions"].append({
                "dataset": entry["dataset"], "session": entry["session"],
                "reloaded_shape": list(entry["counts"].shape),
                "delivered_shape": [delivered["n_trials"], delivered["n_units"], delivered["n_bins"]],
            })
            entries.pop(key)
    reload_check["n_sessions_entering_matched_cells"] = len(entries)
    reload_check["reloaded_counts_reproduce_delivered_shapes"] = not reload_check["mismatched_sessions"]
    print(f"reloaded {reload_check['n_sessions_reloaded']} sessions, "
          f"{reload_check['n_sessions_entering_matched_cells']} usable", file=sys.stderr, flush=True)

    per_cell = run_matched_cells(entries, common_lags, checkpoint, flush)
    _write_json(CHECKPOINT_PATH, checkpoint)
    matched_summary = summarise_matched_cells(per_cell)

    # ── the human regional ladder ──
    ladder_candidates = [row for row in lag_artifact["human_lag_rows"]
                         if row["epoch"] == EPOCH and row["width_bins"] == WIDTH_BINS and row["bin_ms"] == BIN_MS]
    fitted_candidates = [row for row in ladder_candidates
                         if row["profile"].get("status") == "fitted" and row.get("null_permutation") is not None]
    ladder_lags = sorted(set.intersection(*(reachable_lags(row) for row in fitted_candidates)))
    ladder, ladder_dropped = ladder_rows(lag_artifact["human_lag_rows"], ladder_lags)
    descriptives = structure_descriptives(ladder)
    heterogeneity = within_session_heterogeneity(ladder, ("regional_heterogeneity",))
    contrasts = structure_versus_own_pooled(ladder, ("regional_contrast",))
    regional = regional_verdict(heterogeneity, contrasts)

    # The same ladder over the lag range the animal arms also reach, so the human
    # regional numbers can be put beside an animal amplitude without changing which
    # lags each side averaged.
    animal_common = sorted(set(ladder_lags) & set(common_lags) & (set(macaque_lags) if macaque_lags else set(common_lags)))
    ladder_animal_range, _ = ladder_rows(lag_artifact["human_lag_rows"], animal_common) if animal_common else ([], [])
    descriptives_animal_range = structure_descriptives(ladder_animal_range) if ladder_animal_range else {}

    detectability = human_arm_detectability(
        human_amplitudes, mouse_amplitudes, macaque_amplitudes if len(macaque_amplitudes) else None)

    output = {
        "version": "1.0.0",
        "question": ("how much of the shared cross-unit component does each recording preparation deliver to "
                     "this project's split-half persistence estimator, once unit count, trial count, firing "
                     "rate, window length, bin width and estimator width are all held fixed"),
        "scope": {
            "corpora": {
                "human_delay_arm": {"datasets": matching["human_arm"]["datasets"],
                                     "n_sessions": matching["human_arm"]["n_sessions"],
                                     "n_patients": matching["human_arm"]["n_patients"]},
                "mouse_arm": {"datasets": matching["mouse_arm"]["datasets"],
                               "n_sessions": matching["mouse_arm"]["n_sessions"]},
                "macaque_arm": {"n_sessions": macaque_context["n_sessions"],
                                 "role": "scale reference only, not window-matched"},
            },
            "unit_of_analysis": ("one recording session's pooled-structure row for the preparation contrast; "
                                  "one session-by-structure row, nested in session and in patient, for the "
                                  "regional ladder"),
            "amplitude_definition": ("mean over the common reachable lag bins of the observed split-half "
                                      "correlation median minus the per-unit permutation null's median "
                                      "correlation at the same lag"),
            "parameters": {
                "width_bins": WIDTH_BINS, "bin_width_s": BIN_WIDTH_S, "window_s": WINDOW_S,
                "alpha": ALPHA, "n_bootstrap": N_BOOT, "n_permutations": N_PERM,
                "n_subsample_repeats_per_cell": N_SUBSAMPLE_REPEATS,
                "matched_cells_units_by_trials": [list(cell) for cell in MATCHED_CELLS],
                "minimum_sessions_per_arm_per_cell": MIN_SESSIONS_PER_ARM_PER_CELL,
                "minimum_sessions_for_a_structure_to_enter_tests": MIN_LADDER_SESSIONS,
                "material_amplitude_difference": MATERIAL_AMPLITUDE_DIFFERENCE,
            },
            "seed": SEED,
            "exclusions": {
                "human_rows_seen_at_this_width_and_epoch_and_structure":
                    len(human_rows) + len(human_dropped),
                "human_rows_kept": len(human_rows),
                "human_rows_dropped": human_dropped,
                "mouse_rows_seen_at_this_width_and_epoch_and_structure":
                    len(mouse_rows) + len(mouse_dropped),
                "mouse_rows_kept": len(mouse_rows),
                "mouse_rows_dropped": mouse_dropped,
                "sessions_not_reloadable_for_the_matched_cells": human_load_notes + mouse_load_notes,
                "regional_ladder_rows_seen": len(ladder) + len(ladder_dropped),
                "regional_ladder_rows_kept": len(ladder),
                "regional_ladder_rows_dropped": ladder_dropped,
            },
            "checkpoint_path": str(CHECKPOINT_PATH.relative_to(REPO_ROOT)),
            "checkpoint_entries_reused_from_a_previous_run": n_at_start,
            "checkpoint_entries_total": len(checkpoint),
        },
        "estimator_matching_verification": matching,
        "reachability": lag_reachability_note(),
        "unmatched_preparation_contrast": {
            "reference_arm": HUMAN_ARM_LABEL,
            "comparison_arm": MOUSE_ARM_LABEL,
            "lag_range_bins": [common_lags[0], common_lags[-1]],
            **unmatched,
            "reading": ("the ratio of medians is the headline a reader reaches for and is the least "
                        "trustworthy number here, because the human median sits near zero; the difference "
                        "of medians with its interval, and the overlap block, are what the claim rests on"),
        },
        "instrumentally_matched_contrast": {
            "how_matched": ("units and trials are drawn without replacement from each session's own raw "
                            "counts down to a common cell size and the identical estimator is re-run, "
                            f"averaged over {N_SUBSAMPLE_REPEATS} independent draws per session per cell -- not by "
                            "restricting each arm to a band of its own sessions, which discards most of the "
                            "human arm and leaves the surviving distributions unmatched"),
            "reloaded_counts_check": reload_check,
            "per_cell": per_cell,
            "summary": matched_summary,
        },
        "preparation_contrast_verdict": {
            **{k: v for k, v in matched_summary.items() if k in
               ("branch", "branch_meaning", "deciding_cell", "effect_size_at_deciding_cell",
                "n_cells_clearing_fdr", "all_cells_agree_in_sign")},
            "unmatched_effect_size_for_comparison": {
                "human_median_amplitude": unmatched.get("reference_median_amplitude"),
                "mouse_median_amplitude": unmatched.get("comparison_median_amplitude"),
                "median_difference": unmatched.get("median_difference_comparison_minus_reference"),
                "median_difference_ci_95": unmatched.get("median_difference_ci_95"),
                "ratio_of_medians": unmatched.get("ratio_of_medians"),
            },
            "what_this_verdict_may_not_be_read_as": (
                "not a species difference and not a statement that human working memory lacks a shared "
                "population state. Region, task and electrode are all confounded with this comparison and "
                "cannot be matched with these data. The measured quantity is what a preparation delivers "
                "to this estimator."),
            "not_matched_by_this_comparison": NOT_MATCHED_BY_THIS_COMPARISON,
        },
        "human_regional_amplitude_ladder": {
            "lag_range_bins": [ladder_lags[0], ladder_lags[-1]] if ladder_lags else None,
            "lag_range_s": [ladder_lags[0] * BIN_WIDTH_S, ladder_lags[-1] * BIN_WIDTH_S] if ladder_lags else None,
            "why_this_ladder_is_interpretable": ("task, electrode technology, estimator and species are all "
                                                  "fixed across its rungs; only which units are pooled changes"),
            "nesting": ("each session contributes one row per structure it recorded from plus a pooled row "
                        "over all of them, so structure rows are nested in sessions and sessions in patients; "
                        "the heterogeneity null shuffles structure labels within session and every "
                        "structure-level contrast is reported patient-clustered beside its session-level form"),
            "by_structure": descriptives,
            "by_structure_over_the_lag_range_the_animal_arms_also_reach": {
                "lag_range_bins": [animal_common[0], animal_common[-1]] if animal_common else None,
                "by_structure": descriptives_animal_range,
            },
            "between_structure_heterogeneity": heterogeneity,
            "structure_against_own_session_pooled": contrasts,
        },
        "regional_verdict": regional,
        "macaque_scale_reference": macaque_context,
        "human_arm_detectability": detectability,
        "wall_clock_s": time.time() - t_start,
    }

    _write_json(OUTPUT_PATH, output)
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(canonical_json({
        "preparation_branch": matched_summary.get("branch"),
        "deciding_cell": matched_summary.get("deciding_cell"),
        "deciding_cell_effect": matched_summary.get("effect_size_at_deciding_cell"),
        "unmatched_human_median": unmatched.get("reference_median_amplitude"),
        "unmatched_mouse_median": unmatched.get("comparison_median_amplitude"),
        "unmatched_ratio": unmatched.get("ratio_of_medians"),
        "unmatched_difference_ci_95": unmatched.get("median_difference_ci_95"),
        "probability_a_mouse_session_exceeds_a_human_session":
            unmatched.get("overlap", {}).get("probability_comparison_exceeds_reference"),
        "regional_branch": regional.get("branch"),
        "regional_p_value": regional.get("within_session_label_permutation_p_value"),
        "highest_structure": regional.get("highest_structure"),
        "highest_structure_effect": regional.get("highest_structure_effect"),
        "human_minimum_detectable_amplitude":
            detectability["minimum_detectable_amplitude_at_80_percent_power"].get("mdd"),
        "human_sessions_needed_for_its_own_amplitude":
            detectability["power_and_required_sessions_against"]["human_arm_own_mean_amplitude"][
                "human_sessions_needed_for_80_percent_power"],
    }))


if __name__ == "__main__":
    main()
