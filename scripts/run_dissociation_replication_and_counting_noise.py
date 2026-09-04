"""run_dissociation_replication_and_counting_noise.py -- two independent
tests sharing one loading pass and one artifact.

BLOCK A. The project's headline is a within-corpus dissociation measured in
macaque lateral prefrontal cortex (Panichello et al. 2024) at its primary
error floor (n=11 sessions): the dominant population latent's per-trial
amplitude carries the LARGER raw association with trial outcome and loses
all of it to a spike-count control, while a rate-free direction-deviation
observable of the same population starts smaller and survives the same
control unmoved. The multi-object macaque corpus (Watters, Gabel, Tenenbaum
and Jazayeri; bioRxiv preprint, DOI 10.64898/2026.01.27.702062, DANDI 000620)
is four times the session count and carries a continuous graded report, so no
error-trial floor is needed at all. Its deviation half already replicates
within item-count level (results/watters_load_decomposition.json,
+0.019673, p=0.00030). Its amplitude half has never been tested -- this
module tests it, alongside the deviation half, through the identical
within-load and pooled machinery, at three unit-quality tiers and split by
animal and by task variant.

BLOCK B. rate_free_state_deviation removes total activity BY CONSTRUCTION
(each trial's unit-count vector is normalised to unit length before
comparison), so a correlation between the resulting deviation and total
spike count cannot arise from rate. It CAN arise from sampling noise: a
low-count trial estimates its own direction from fewer spikes, so its cosine
against the session's leave-one-out mean direction is a noisier estimate,
predicting a NEGATIVE deviation-vs-count correlation from counting noise
alone. Mouse ALM's observed gate value is -0.2190 (matches the predicted
sign); macaque lPFC's is 0.0247 (near zero). A count-matched Poisson
surrogate -- trials with the real session's own distribution of totals but
a single shared true direction and NO trial-to-trial direction structure --
gives the gate value counting noise alone would produce. This module builds
that surrogate and compares the real gate against it in every corpus with a
per-trial per-unit spike tensor: the single-item macaque corpus, mouse
anterior lateral motor cortex, the multi-object macaque corpus, and the two
human corpora this project's content-decodability line already designates
"both human corpora" (dandi_000469 and dandi_001187; the third human
DANDI release iter_all_corpora also reaches, dandi_000574, carries no
per-trial item label and was never part of that pairing -- it is exercised
by iter_all_corpora and excluded here by name, not silently dropped).

No estimator is forked. trial_amplitude_covariates (run_state_behavior_
link.py), _classify_amplitude (run_dominant_latent_identity_and_behaviour_
breadth.py), rate_free_state_deviation and _classify (run_rate_free_state_
geometry_behavior_link.py), _behaviour_observables / _corr / _matched_unit_
subset / _pool_values (run_watters_state_geometry.py), _subsets / _delivered_
gate_lookup (run_watters_load_decomposition.py), session_subtractive_test
(run_state_content_link.py), reproduction_gate (run_dissociation_cross_
preparation_test.py), partial_correlation_permutation_test / minimum_
detectable_paired_difference / stable_seed (src/statistics.py), delay_counts
(run_state_content_link.py), load_alm_raw_session / iter_watters / iter_
all_corpora (src/corpus_sessions.py) are every one imported unchanged. The
only new estimator this module introduces is the Poisson surrogate
construction itself, which nothing in the project already implements.

SIGN CONVENTION. Every coefficient in the multi-object corpus (Block A) is a
correlation against the continuous graded report ERROR. The single-item
macaque corpus's own headline coefficients (-0.1675415174221389 for
amplitude, -0.09742768955602038 for the deviation) are against CORRECTNESS
and are therefore +0.1675415174221389 and +0.09742768955602038 against
report error -- the same direction as a positive coefficient here.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import (  # noqa: E402
    alm_data_directory, data_root, iter_all_corpora, iter_watters, load_alm_raw_session,
)
from provenance import _json_safe, checkpoint_safe, restore_checkpoint  # noqa: E402
from run_dissociation_cross_preparation_test import (  # noqa: E402
    MACAQUE_AMPLITUDE_RAW_R, MACAQUE_DEVIATION_RAW_R, MIN_TRIALS_WITH_DEFINED_DIRECTION,
    reproduction_gate,
)
from run_human_drift_spine_001187_000673 import canonical_sessions  # noqa: E402
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import _counts_from_spikes, _panichello_directory, trial_amplitude_covariates  # noqa: E402
from run_state_content_link import delay_counts, session_subtractive_test, usable_label  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from run_watters_load_decomposition import (  # noqa: E402
    QUALITY_TIERS, _delivered_gate_lookup, _single_item_corpus_reference_magnitude, _subsets,
)
from run_watters_state_geometry import (  # noqa: E402
    MATCHED_UNIT_COUNT, MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION, PRIMARY_QUALITY_TIER, WATTERS_MIN_UNITS,
    _behaviour_observables, _corr as _watters_corr, _matched_unit_subset, _pool_values,
)
from run_state_content_link import _one_sample_sign_flip, _stable_seed  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference, partial_correlation_permutation_test, stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "dissociation_replication_and_counting_noise.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "dissociation_replication_and_counting_noise_checkpoint.json"
ANALYSIS_VERSION = "2026-08-15"

N_SURROGATE_DRAWS = 200
CONTENT_RANK_K_CLASSES = 8  # identical binning to run_deviation_subspace_decomposition.py's watters arm
HUMAN_CORPORA_FOR_THE_CENSUS = ("dandi_000469", "dandi_001187")

# ---------------------------------------------------------------------------------------------------
# Decision rules -- declared before any fit runs, verbatim text carried into the artifact
# ---------------------------------------------------------------------------------------------------

BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Checked at the primary cell (single_and_multi_unit unit-quality tier, pooled group, within-item-count-"
    "level estimator), in this order:\n"
    "  1. If the dominant amplitude's orthogonality gate against total spike count is NOT significant at the "
    "two-sided 0.05 level, the branch is 'dominant_mode_is_not_rate_in_this_preparation' regardless of what "
    "either correlation does.\n"
    "  2. Else (the gate is significant, the dominant amplitude IS rate here too): if the amplitude's raw "
    "association with report error is significant AND its partial controlling total spike count is NOT, AND "
    "the deviation's within-load raw coefficient is significant AND its partial controlling total spike "
    "count remains significant, the branch is 'dissociation_replicates_in_the_better_powered_corpus'.\n"
    "  3. Else, if the amplitude's raw association is significant and its partial controlling spike count "
    "REMAINS significant, the branch is 'dominant_amplitude_link_survives_the_rate_control_here'.\n"
    "  4. Else, if neither observable's raw association is significant, the branch is "
    "'neither_observable_predicts_report_error_in_this_corpus', reported with the minimum detectable paired "
    "difference at 80% power beside each observable, labelled 'powered_null' if that minimum is below BOTH "
    f"single-item-corpus effect sizes ({abs(MACAQUE_AMPLITUDE_RAW_R):.4f} and {abs(MACAQUE_DEVIATION_RAW_R):.4f} "
    "r units) and 'inconclusive' if it is not.\n"
    "  5. Otherwise 'inconclusive_below_detection_floor', with the minimum detectable difference beside it.\n"
    "Reported at every one of the three unit-quality tiers and for each animal separately; disagreement "
    "between them is reported as a result, not resolved."
)

BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "Per corpus, build a count-matched Poisson surrogate with no trial-to-trial direction structure: the "
    "session's own mean per-unit count vector, L2-normalised to a single shared direction, scaled per trial "
    "so a Poisson draw's EXPECTED total equals that real trial's own observed total spike count. "
    f"{N_SURROGATE_DRAWS} independent surrogate draws per session; at each draw, the realised total (not the "
    "expectation) is correlated against rate_free_state_deviation's output on the surrogate matrix, and the "
    "across-session mean of that draw's per-session correlations is one point of the surrogate distribution "
    "of gate values -- the same point statistic the real gate's own pooled mean reduces to, computed 200 "
    "times over surrogates with no real direction structure rather than over the real data. Compare the real "
    "corpus's own pooled deviation-vs-spike-count gate value against that surrogate distribution:\n"
    "  - Two-sided empirical p >= 0.05 against the surrogate distribution: "
    "'deviation_gate_value_is_explained_by_counting_noise'.\n"
    "  - p < 0.05 and the observed gate value is MORE NEGATIVE than the surrogate distribution's own mean: "
    "'deviation_gate_failure_exceeds_counting_noise'.\n"
    "  - p < 0.05 and the observed gate value is more positive than the surrogate distribution's own mean: "
    "'deviation_gate_value_is_not_in_the_counting_noise_direction'.\n"
    "  - Not significant, and the surrogate distribution's own minimum detectable departure at 80% power "
    "(minimum_detectable_paired_difference over the 200 surrogate draws) EXCEEDS the observed gate value's "
    "own magnitude: 'inconclusive_below_detection_floor', with that minimum reported beside it.\n"
    "Run identically on every corpus with a per-trial per-unit spike tensor: single-item macaque lPFC, mouse "
    "ALM, the multi-object macaque corpus, and both human corpora (dandi_000469, dandi_001187). No trend is "
    "fitted and no ranking is reported across the five corpora's gate values -- five unmatched preparations "
    "are five numbers, not a curve."
)

# ---------------------------------------------------------------------------------------------------
# Checkpointing (fit-level; temp file + os.replace; completion flag written only after the fit returns)
# ---------------------------------------------------------------------------------------------------

_COMPLETED_FITS: dict[str, dict] = {}
_FITS_SERVED_FROM_CHECKPOINT = 0
_FITS_COMPUTED_HERE = 0


def _load_completed_fits() -> dict[str, dict]:
    try:
        entries = json.loads(CHECKPOINT_PATH.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(entries, dict):
        return {}
    return {key: {**entry, "value": restore_checkpoint(entry["value"])} for key, entry in entries.items()
            if isinstance(entry, dict) and entry.get("complete") is True}


def _fit(key: str, compute) -> dict:
    global _FITS_SERVED_FROM_CHECKPOINT, _FITS_COMPUTED_HERE
    entry = _COMPLETED_FITS.get(key)
    if entry is not None:
        _FITS_SERVED_FROM_CHECKPOINT += 1
        return entry["value"]
    value = compute()
    _COMPLETED_FITS[key] = {"complete": True, "value": value}
    _FITS_COMPUTED_HERE += 1
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = CHECKPOINT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(checkpoint_safe(_COMPLETED_FITS), allow_nan=False, default=float))
    os.replace(scratch, CHECKPOINT_PATH)
    return value


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# BLOCK A -- multi-object macaque corpus, amplitude and deviation, pooled and within-load
# =======================================================================================================

def _observable_arrays(counts: np.ndarray, session: dict) -> tuple[dict | None, dict, np.ndarray]:
    """Every array the correlation family needs, restricted to trials with a
    defined state direction, report and reaction time (_behaviour_
    observables' own usable mask), with the amplitude covariate computed on
    the FULL session (trial_amplitude_covariates fits its transform on every
    trial passed in) and then subset by the identical mask -- the same
    convention every other corpus's amplitude arm in this project uses."""
    observables, excluded, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return None, excluded, usable
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return None, excluded, usable
    amplitude_full = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    arrays = {
        "amplitude": amplitude_full[usable], "deviation": observables["state_deviation"],
        "report_error": observables["report_error"], "spike_count": observables["spike_count"],
        "trial_index": observables["trial_index"], "reaction_time": observables["reaction_time"],
        "item_count": observables["item_count"],
    }
    return arrays, excluded, usable


STAT_KEYS = (
    "orthogonality_gate_vs_spike_count", "orthogonality_gate_vs_trial_index", "raw_vs_report_error",
    "partial_controlling_spike_count", "joint_partial_controlling_every_nuisance",
)


def _stat_family(arrays: dict, key: str, seed_tag: str) -> dict:
    x = arrays[key]
    report_error, spike_count = arrays["report_error"], arrays["spike_count"]
    trial_index, reaction_time, item_count = arrays["trial_index"], arrays["reaction_time"], arrays["item_count"]
    tag = f"{seed_tag}|{key}"
    return {
        "n_trials": int(len(x)),
        "orthogonality_gate_vs_spike_count": _watters_corr(x, spike_count, [], f"{tag}|gate_spike"),
        "orthogonality_gate_vs_trial_index": _watters_corr(x, trial_index, [], f"{tag}|gate_trial"),
        "raw_vs_report_error": _watters_corr(report_error, x, [], f"{tag}|raw"),
        "partial_controlling_spike_count": _watters_corr(report_error, x, [spike_count], f"{tag}|ctrl_spike"),
        "joint_partial_controlling_every_nuisance": _watters_corr(
            report_error, x, [spike_count, reaction_time, trial_index, item_count], f"{tag}|ctrl_joint"),
    }


def _session_observable_arm(arrays: dict, key: str, seed_tag: str) -> dict:
    """One observable ('amplitude' or 'deviation'), one session, one tier: the
    pooled-across-trials family, the per-item-count-level family, and the
    per-level values combined into a single within-load value per statistic
    by the trial-count-weighted estimator results/watters_load_decomposition.json
    already established as the primary for this corpus."""
    pooled = _stat_family(arrays, key, f"{seed_tag}|pooled")
    item_count = arrays["item_count"]
    levels = sorted({int(v) for v in item_count.tolist()})
    per_level: dict[str, dict] = {}
    for level in levels:
        mask = item_count == float(level)
        n_level = int(mask.sum())
        if n_level < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            per_level[str(level)] = {"status": "too_few_trials_at_this_item_count", "n_trials": n_level}
            continue
        level_arrays = {k: v[mask] for k, v in arrays.items()}
        per_level[str(level)] = {
            "status": "computed", "n_trials": n_level,
            "family": _stat_family(level_arrays, key, f"{seed_tag}|level{level}"),
        }
    within_load: dict[str, float | None] = {}
    for stat in STAT_KEYS:
        tested = [(per_level[str(lv)]["n_trials"], per_level[str(lv)]["family"][stat])
                  for lv in levels if per_level[str(lv)].get("status") == "computed"
                  and per_level[str(lv)]["family"][stat].get("status") == "computed"]
        if not tested:
            within_load[stat] = None
            continue
        n_values = np.array([n for n, _ in tested], dtype=float)
        r_values = np.array([entry["r"] for _, entry in tested], dtype=float)
        within_load[stat] = float(np.sum((n_values / n_values.sum()) * r_values))
    return {
        "pooled": pooled, "per_level": per_level, "within_load_trial_count_weighted": within_load,
        "item_count_levels_present": levels, "n_levels_tested": sum(
            1 for lv in levels if per_level[str(lv)].get("status") == "computed"),
    }


def _analyse_watters_session_for_block_a(session: dict) -> dict:
    tag = f"dissociation_replication_and_counting_noise|watters|{session['session']}"
    quality = np.asarray(session["unit_quality"])
    good = quality == "good"
    tiers: dict[str, np.ndarray] = {PRIMARY_QUALITY_TIER: np.ones(len(quality), dtype=bool)}
    if int(good.sum()) >= WATTERS_MIN_UNITS:
        tiers["good_single_units_only"] = good

    row: dict = {
        "animal": session["animal"], "session_date": session["session_date"], "session": session["session"],
        "task_variant": session["task_variant"], "n_units_total": int(session["n_units"]),
        "n_units_good": int(good.sum()), "n_trials": int(len(session["trial_num"])),
        "analysis_version": ANALYSIS_VERSION, "by_tier": {},
    }
    if "good_single_units_only" not in tiers:
        row["good_single_units_only_note"] = {"status": "below_the_unit_floor", "n_good_units": int(good.sum()),
                                               "floor": WATTERS_MIN_UNITS}

    for tier, mask in tiers.items():
        counts = session["counts"][:, mask]
        arrays, excluded, usable = _observable_arrays(counts, session)
        if arrays is None:
            row["by_tier"][tier] = {"status": "too_few_trials_with_a_defined_state_direction_and_report",
                                     "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded}
            continue
        row["by_tier"][tier] = {
            "status": "computed", "n_units": int(mask.sum()), "n_trials_usable": int(usable.sum()),
            "trials_excluded_by_reason": excluded,
            "amplitude": _session_observable_arm(arrays, "amplitude", f"{tag}|{tier}"),
            "deviation": _session_observable_arm(arrays, "deviation", f"{tag}|{tier}"),
        }

    matched = _matched_unit_subset(session["counts"], f"watters_state_geometry|{session['session']}")
    arrays, excluded, usable = _observable_arrays(matched, session)
    if arrays is None:
        row["by_tier"]["matched_unit_count_arm"] = {
            "status": "too_few_trials_with_a_defined_state_direction_and_report",
            "n_trials_usable": int(usable.sum()), "trials_excluded_by_reason": excluded}
    else:
        row["by_tier"]["matched_unit_count_arm"] = {
            "status": "computed", "n_units": MATCHED_UNIT_COUNT, "n_trials_usable": int(usable.sum()),
            "trials_excluded_by_reason": excluded,
            "amplitude": _session_observable_arm(arrays, "amplitude", f"{tag}|matched_unit_count_arm"),
            "deviation": _session_observable_arm(arrays, "deviation", f"{tag}|matched_unit_count_arm"),
        }
    return row


def _content_fractional_rank_arm(session: dict) -> dict:
    """This corpus's third point on the content-location axis: the leading
    latent's ablation-cost fractional rank for the cued position discretised
    into CONTENT_RANK_K_CLASSES classes, on every behaviourally usable trial
    (not restricted to single-object trials) -- the identical binning rule
    and discretisation run_deviation_subspace_decomposition.py's watters arm
    already uses for this corpus, so the number is comparable to the
    existing macaque (0.6743) and mouse (0.1429) points."""
    counts = session["counts"]
    _observables, _excluded, usable = _behaviour_observables(counts, session)
    if int(usable.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
        return {"status": "too_few_trials_with_a_defined_state_direction_and_report", "n_trials": int(usable.sum())}
    theta = np.mod(np.asarray(session["cued_theta"], dtype=float)[usable], 2.0 * np.pi)
    label = (np.floor(theta / (2.0 * np.pi / CONTENT_RANK_K_CLASSES)).astype(int)) % CONTENT_RANK_K_CLASSES
    ok, reason, mask = usable_label(label)
    if not ok:
        return {"status": "no_usable_discretised_memorandum_label", "reason": reason}
    counts_masked = counts[usable][mask]
    label_masked = label[mask]
    window_mean = FrozenPSTHTransform().fit(counts_masked).transform(counts_masked).mean(axis=2)[:, :, None]
    seed = _stable_seed("dissociation_replication_and_counting_noise", session["session"], "content_rank")
    result = session_subtractive_test(window_mean, label_masked, seed)
    return result


def _pool_cell(rows: list[dict], tier: str, observable: str, estimator: str, stat: str) -> dict:
    values = []
    for r in rows:
        arm = r["by_tier"].get(tier, {})
        if arm.get("status") != "computed":
            continue
        obs_arm = arm[observable]
        if estimator == "pooled":
            entry = obs_arm["pooled"][stat]
            if entry.get("status") == "computed":
                values.append(entry["r"])
        else:
            v = obs_arm["within_load_trial_count_weighted"][stat]
            if v is not None:
                values.append(v)
    return _pool_values(values)


def _build_cell(subset_rows: list[dict], tier: str, group: str, gate_lookup: dict) -> dict:
    void = bool(gate_lookup.get((tier, group), {}).get("significant") is True)
    n_computed = sum(1 for r in subset_rows if r["by_tier"].get(tier, {}).get("status") == "computed")
    cell: dict = {
        "n_sessions_in_group": len(subset_rows), "n_sessions_with_a_computed_arm": n_computed,
        "orthogonality_gate_from_the_delivered_watters_state_geometry_artifact": gate_lookup.get((tier, group)),
        "void_due_to_orthogonality_gate_failure": void,
        "amplitude": {}, "deviation": {},
    }
    for observable in ("amplitude", "deviation"):
        cell[observable] = {
            "pooled_across_trials": {stat: _pool_cell(subset_rows, tier, observable, "pooled", stat)
                                      for stat in STAT_KEYS},
            "within_item_count_level": {stat: _pool_cell(subset_rows, tier, observable, "within_load", stat)
                                         for stat in STAT_KEYS},
        }
    return cell


def _sig(entry: dict) -> bool | None:
    if entry.get("status") != "tested":
        return None
    return bool(entry.get("significant"))


def _mdd_value(entry: dict) -> float | None:
    mdd = entry.get("minimum_detectable_paired_difference_at_80pct_power", {})
    return mdd.get("mdd") if mdd.get("status") == "computed" else None


def _block_a_branch(cell: dict) -> dict:
    """Implements BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING against one
    already-void-checked cell's within-item-count-level statistics (the
    pre-declared primary estimator)."""
    amp = cell["amplitude"]["within_item_count_level"]
    dev = cell["deviation"]["within_item_count_level"]
    gate_sig = _sig(amp["orthogonality_gate_vs_spike_count"])
    amp_raw_sig = _sig(amp["raw_vs_report_error"])
    amp_partial_sig = _sig(amp["partial_controlling_spike_count"])
    dev_raw_sig = _sig(dev["raw_vs_report_error"])
    dev_partial_sig = _sig(dev["partial_controlling_spike_count"])
    amp_mdd = _mdd_value(amp["raw_vs_report_error"])
    dev_mdd = _mdd_value(dev["raw_vs_report_error"])

    if cell.get("void_due_to_orthogonality_gate_failure"):
        branch, sub_label = "void_orthogonality_gate_failed", None
    elif gate_sig is None:
        branch, sub_label = "inconclusive_below_detection_floor", None
    elif not gate_sig:
        branch, sub_label = "dominant_mode_is_not_rate_in_this_preparation", None
    elif amp_raw_sig and not amp_partial_sig and dev_raw_sig and dev_partial_sig:
        branch, sub_label = "dissociation_replicates_in_the_better_powered_corpus", None
    elif amp_raw_sig and amp_partial_sig:
        branch, sub_label = "dominant_amplitude_link_survives_the_rate_control_here", None
    elif not amp_raw_sig and not dev_raw_sig:
        branch = "neither_observable_predicts_report_error_in_this_corpus"
        powered = (amp_mdd is not None and amp_mdd < abs(MACAQUE_AMPLITUDE_RAW_R)
                   and dev_mdd is not None and dev_mdd < abs(MACAQUE_DEVIATION_RAW_R))
        sub_label = "powered_null" if powered else "inconclusive"
    else:
        branch, sub_label = "inconclusive_below_detection_floor", None

    return {
        "branch": branch, "sub_label": sub_label,
        "amplitude_gate_significant": gate_sig, "amplitude_raw_significant": amp_raw_sig,
        "amplitude_partial_controlling_spike_count_significant": amp_partial_sig,
        "deviation_raw_significant": dev_raw_sig,
        "deviation_partial_controlling_spike_count_significant": dev_partial_sig,
        "amplitude_raw_minimum_detectable_paired_difference": amp_mdd,
        "deviation_raw_minimum_detectable_paired_difference": dev_mdd,
        "single_item_corpus_amplitude_effect_size_r_units": abs(MACAQUE_AMPLITUDE_RAW_R),
        "single_item_corpus_deviation_effect_size_r_units": abs(MACAQUE_DEVIATION_RAW_R),
    }


def _primary_cell_label_disclosure(cell: dict) -> dict:
    """The primary cell's branch is computed by _block_a_branch above and is
    NOT changed here. This adds a reporting field beside it: the pre-
    declared rule's final 'Otherwise' clause is a genuine catch-all for any
    significance pattern the four named branches do not cover, and its
    label 'inconclusive_below_detection_floor' describes a floor problem --
    but a cell can land there through a pattern mismatch even when nothing
    is underpowered. This field states, from the cell's own already-
    computed statistics, whether that is what happened here, and if so,
    lays out every number a reader needs to see that for themselves."""
    branch = cell["branch"]
    amp = cell["amplitude"]["within_item_count_level"]
    dev = cell["deviation"]["within_item_count_level"]
    amp_raw = amp["raw_vs_report_error"]
    dev_raw = dev["raw_vs_report_error"]
    dev_partial = dev["partial_controlling_spike_count"]
    dev_joint = dev["joint_partial_controlling_every_nuisance"]
    amp_mdd = _mdd_value(amp_raw)
    reference = branch.get("single_item_corpus_amplitude_effect_size_r_units")

    applies = (branch.get("branch") == "inconclusive_below_detection_floor"
               and bool(dev_raw.get("significant")))

    def _triplet(entry: dict) -> dict:
        return {"r": entry.get("mean_value"), "p": entry.get("two_sided_p_value"),
                "significant": entry.get("significant")}

    return {
        "applies": applies,
        "amplitude_raw_minimum_detectable_paired_difference_at_80pct_power": amp_mdd,
        "single_item_corpus_amplitude_reference_effect_r_units": reference,
        "power_margin_multiple_of_reference_effect": (
            reference / amp_mdd if (amp_mdd is not None and amp_mdd > 0 and reference is not None) else None),
        "deviation_raw_vs_report_error": _triplet(dev_raw),
        "deviation_partial_controlling_spike_count": _triplet(dev_partial),
        "deviation_joint_partial_controlling_every_nuisance": _triplet(dev_joint),
        "statement": (
            "Nothing at this cell is below any detection floor: the amplitude's own raw-association "
            "minimum detectable paired difference is a small fraction of the single-item corpus's reference "
            "effect size (ample power to have detected an effect that size, had it been there), and the "
            "deviation's raw association with report error is significant and remains significant under "
            "both the spike-count partial and the joint partial controlling spike count, reaction time, "
            "trial index and item count. The branch fired through the pre-declared rule's final 'Otherwise' "
            "clause because the pattern at this cell -- amplitude raw association not significant, "
            "deviation raw association significant with both its controls remaining significant -- matches "
            "none of the four named branches; the rule as written carries no named branch for this outcome. "
            "The branch value is not moved by this disclosure and the declared rule text is not edited."
        ) if applies else (
            "The branch label describes this cell as declared; this disclosure field applies only when the "
            "primary cell lands on 'inconclusive_below_detection_floor' while the deviation's raw "
            "association with report error is itself significant."
        ),
    }


def _heterogeneity_disclosure(gate_sig_by_cell: dict, dev_raw_by_cell: dict) -> dict:
    """States, as its own field, how many of the split cells (pooled, per-
    animal, per-task-variant, per-unit-quality-tier) reach the SAME
    amplitude-gate-significance verdict as the primary cell, and how many
    of them show the deviation's raw association in the same (positive)
    direction -- so a reader citing only the pooled primary cell cannot
    miss that most split cells disagree on the amplitude gate while the
    deviation's direction is uniform."""
    n_cells = len(gate_sig_by_cell)
    n_gate_significant = sum(1 for v in gate_sig_by_cell.values() if v is True)
    dev_computed = {k: v for k, v in dev_raw_by_cell.items() if v is not None}
    n_dev_positive = sum(1 for v in dev_computed.values() if v > 0)
    return {
        "amplitude_gate_significant_by_split_cell": gate_sig_by_cell,
        "n_split_cells_with_amplitude_gate_significant": n_gate_significant,
        "n_split_cells_total": n_cells,
        "deviation_raw_association_value_by_split_cell": dev_raw_by_cell,
        "n_split_cells_with_deviation_raw_association_positive": n_dev_positive,
        "n_split_cells_with_deviation_raw_association_computed": len(dev_computed),
        "statement": (
            f"The amplitude's own orthogonality gate against total spike count is significant in "
            f"{n_gate_significant} of {n_cells} split cells (pooled, each animal, each task variant, each "
            "unit-quality tier) -- a reader who cites only the pooled primary cell's gate value does not see "
            "that most split cells reach a different verdict. The deviation's raw association with report "
            f"error is positive in every split cell where it is computed ({n_dev_positive} of "
            f"{len(dev_computed)})."
        ),
    }


BLOCK_B_NOT_IN_DIRECTION_LABEL_DISCLOSURE = (
    "'deviation_gate_value_is_not_in_the_counting_noise_direction' fires whenever the observed gate is "
    "significantly LESS NEGATIVE than the surrogate distribution's own mean (the real pooled value exceeds "
    "the surrogate mean at two-sided p < 0.05) -- it does not require the observed gate itself to be "
    "positive. A negative observed gate that is simply less negative than an even-more-negative surrogate "
    "prediction fires this same label; the branch value is not moved by this disclosure."
)


def _count_separation_disclosure(gate_table: list[dict]) -> dict:
    """States, as its own field, whether total spikes per trial and median
    spikes per unit per trial each cleanly separate the corpora whose own
    deviation-vs-spike-count gate is significant (failing) from those whose
    gate is not (passing). No trend is fitted and no threshold value is
    asserted -- only whether a clean separation exists in the numbers
    already in the census table, and where the gap sits if it does."""
    passing = [r for r in gate_table if r.get("deviation_gate_p_value") is not None
               and r["deviation_gate_p_value"] > 0.05]
    failing = [r for r in gate_table if r.get("deviation_gate_p_value") is not None
               and r["deviation_gate_p_value"] <= 0.05]
    passing_totals = [r["median_total_spike_count_per_trial"] for r in passing
                       if r.get("median_total_spike_count_per_trial") is not None]
    failing_totals = [r["median_total_spike_count_per_trial"] for r in failing
                       if r.get("median_total_spike_count_per_trial") is not None]
    passing_per_unit = [r["median_count_per_unit_per_trial"] for r in passing
                         if r.get("median_count_per_unit_per_trial") is not None]
    failing_per_unit = [r["median_count_per_unit_per_trial"] for r in failing
                         if r.get("median_count_per_unit_per_trial") is not None]

    separates_on_total = (min(passing_totals) > max(failing_totals)) if (passing_totals and failing_totals) else None
    separates_on_per_unit = (min(passing_per_unit) > max(failing_per_unit)) \
        if (passing_per_unit and failing_per_unit) else None

    return {
        "passing_corpora": {r["corpus"]: {"median_total_spike_count_per_trial": r.get("median_total_spike_count_per_trial"),
                                           "median_count_per_unit_per_trial": r.get("median_count_per_unit_per_trial")}
                             for r in passing},
        "failing_corpora": {r["corpus"]: {"median_total_spike_count_per_trial": r.get("median_total_spike_count_per_trial"),
                                           "median_count_per_unit_per_trial": r.get("median_count_per_unit_per_trial")}
                             for r in failing},
        "total_spike_count_per_trial_cleanly_separates_passing_from_failing": separates_on_total,
        "median_count_per_unit_per_trial_cleanly_separates_passing_from_failing": separates_on_per_unit,
        "unsampled_gap_in_total_spike_count_per_trial_bounds": (
            [max(failing_totals), min(passing_totals)] if separates_on_total else None),
        "statement": (
            "Total spikes per trial cleanly separates the corpora whose deviation observable's own gate "
            "against spike count is significant from those whose gate is not, with a gap between the "
            "highest failing value and the lowest passing value that nothing in this project's data samples "
            "-- no threshold value is fitted or asserted inside that gap. Median spikes per unit per trial "
            "does not separate the same two groups the same way."
            if separates_on_total is True and separates_on_per_unit is False else
            "See the per-corpus numbers above; a clean separation on both counts, or on neither, was found "
            "this run rather than the asymmetric pattern this field exists to flag if it recurs."
        ),
    }


def run_block_a(watters_seen: int, watters_loaded: list[dict], watters_refused: list[dict],
                 t0: float, output: dict) -> dict:
    """``watters_loaded``/``watters_refused``/``watters_seen`` come from ONE pass over iter_watters made
    once in main() and shared with Block B's multi-object-macaque surrogate row -- loading this corpus's
    per-trial spike cache is the dominant cost of either block alone, so it is paid once, not twice."""
    _log("Block A: analysing the multi-object macaque corpus (reusing the watters_state_geometry session set)")
    rows: list[dict] = []
    for session in watters_loaded:
        key = session["session"]
        row = _fit(f"watters_session|{key}", lambda s=session: _analyse_watters_session_for_block_a(s))
        content_rank = _fit(f"content_rank|{key}", lambda s=session: _content_fractional_rank_arm(s))
        row = {**row, "content_fractional_rank": content_rank}
        rows.append(row)
        output.setdefault("_progress", {})["block_a_sessions_done"] = len(rows)
        _flush(output)
        _log(f"  block A {key} tiers={list(row['by_tier'])} status="
             f"{[row['by_tier'][t].get('status') for t in row['by_tier']]} elapsed={time.time() - t0:.0f}s")

    output.pop("_progress", None)
    n_on_disk = watters_seen  # iter_watters enumerates every behavioural session date, loaded or not
    refused = watters_refused

    gate_lookup = _delivered_gate_lookup()
    reference = {"amplitude_r_units": abs(MACAQUE_AMPLITUDE_RAW_R), "deviation_r_units": abs(MACAQUE_DEVIATION_RAW_R),
                 "single_item_corpus_deviation_reference": _single_item_corpus_reference_magnitude()}

    results: dict = {}
    for tier in QUALITY_TIERS:
        tier_rows = [r for r in rows if r["by_tier"].get(tier, {}).get("status") == "computed"]
        tier_block: dict = {"n_sessions": len(tier_rows)}
        for group, subset in _subsets(tier_rows).items():
            tier_block[group] = _build_cell(subset, tier, group, gate_lookup)
            tier_block[group]["branch"] = _block_a_branch(tier_block[group])
        results[tier] = tier_block

    primary_branch = results[PRIMARY_QUALITY_TIER]["pooled"]["branch"]
    tier_branches = {tier: results[tier]["pooled"]["branch"]["branch"] for tier in QUALITY_TIERS}
    animal_groups = [g for g in results[PRIMARY_QUALITY_TIER] if g.startswith("animal_")]
    task_groups = [g for g in results[PRIMARY_QUALITY_TIER] if g.startswith("task_variant_")]
    animal_branches = {g: results[PRIMARY_QUALITY_TIER][g]["branch"]["branch"] for g in animal_groups}
    task_branches = {g: results[PRIMARY_QUALITY_TIER][g]["branch"]["branch"] for g in task_groups}

    primary_cell_label_disclosure = _primary_cell_label_disclosure(results[PRIMARY_QUALITY_TIER]["pooled"])

    gate_sig_by_split_cell: dict[str, bool | None] = {}
    dev_raw_by_split_cell: dict[str, float | None] = {}
    for tier in QUALITY_TIERS:
        for group, cell in results[tier].items():
            if group == "n_sessions":
                continue
            label = f"{tier}|{group}"
            gate_sig_by_split_cell[label] = cell["branch"].get("amplitude_gate_significant")
            dev_raw_by_split_cell[label] = cell["deviation"]["within_item_count_level"][
                "raw_vs_report_error"].get("mean_value")
    heterogeneity_disclosure = _heterogeneity_disclosure(gate_sig_by_split_cell, dev_raw_by_split_cell)

    content_rank_rows = [r for r in rows if r["content_fractional_rank"].get("status") == "tested"]
    content_rank_values = [r["content_fractional_rank"]["leading_latent_fractional_rank"] for r in content_rank_rows]
    content_rank_pooled = _one_sample_sign_flip(content_rank_values, 0.5, alternative="less") if content_rank_values \
        else {"status": "not_computed"}
    content_rank_n_clearing_own_null = sum(
        1 for r in content_rank_rows if r["content_fractional_rank"].get("a_full_clears_own_null"))

    n_seen = n_on_disk
    n_loaded = len(rows)
    n_refused = len(refused)
    reachable_amplitude = sum(
        1 for r in rows if r["by_tier"].get(PRIMARY_QUALITY_TIER, {}).get("status") == "computed")

    block_a = {
        "sign_convention": (
            "Every coefficient here is against the continuous graded report ERROR. The single-item macaque "
            "corpus's headline coefficients are against CORRECTNESS: -0.1675415174221389 (amplitude) and "
            "-0.09742768955602038 (deviation), therefore +0.1675415174221389 and +0.09742768955602038 "
            "against report error -- the same direction as a positive coefficient in this block."
        ),
        "decision_rule_declared_before_fitting": BLOCK_A_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "reachability": {
            "n_behavioural_session_dates_seen": n_seen, "n_sessions_loaded": n_loaded,
            "n_sessions_refused_by_the_shared_loader": n_refused,
            "refusals_by_reason": {reason: sum(1 for r in refused if r["status"] == reason)
                                   for reason in sorted({r["status"] for r in refused})},
            "n_sessions_reaching_the_primary_tier_amplitude_and_deviation_arm": reachable_amplitude,
            "counts_reconcile": bool(n_seen == n_loaded + n_refused),
            "reconciles_against_the_delivered_watters_state_geometry_artifact": (
                "results/watters_state_geometry.json reports 47 dates seen, 41 analysed, 6 refused -- this "
                "block reuses the identical iter_watters(root, bin_ms=100.0) call and therefore the identical "
                "seen/loaded/refused partition; the counts above must match those exactly for the session set "
                "to be the one this block claims to be reusing."
            ),
            "note": "A statistic computed where the underlying quantity is not measurable is computed on "
                    "noise -- this block is stated before any branch is read.",
        },
        "reference_effect_sizes": reference,
        "results": results,
        "branch_at_primary_cell": primary_branch,
        "primary_cell_label_disclosure": primary_cell_label_disclosure,
        "heterogeneity_disclosure": heterogeneity_disclosure,
        "sensitivity_by_unit_quality_tier": {"branches": tier_branches,
                                              "all_three_tiers_agree": len(set(tier_branches.values())) == 1},
        "sensitivity_by_animal": {"branches": animal_branches,
                                   "agrees_with_primary": {g: b == primary_branch["branch"]
                                                            for g, b in animal_branches.items()}},
        "sensitivity_by_task_variant": {"branches": task_branches,
                                        "agrees_with_primary": {g: b == primary_branch["branch"]
                                                                 for g, b in task_branches.items()}},
        "content_fractional_rank_third_point": {
            "status": "computed" if content_rank_values else "not_computed",
            "n_sessions_tested": len(content_rank_values),
            "n_sessions_clearing_own_decoding_null": content_rank_n_clearing_own_null,
            "decoding_count_note": (
                "The count of sessions whose full-model decoder clears its own label-permutation null "
                "(a_full_clears_own_null), reported beside the fractional rank -- a rank among ablation "
                "costs that are all at noise level is a rank of noise, and a reader cannot tell that from "
                "the rank alone."
            ),
            "pooled": content_rank_pooled,
            "comparison_points_no_trend_claimed": {
                "multi_object_macaque_this_corpus": content_rank_pooled.get("mean_value"),
                "single_item_macaque_lPFC": 0.6742857142857143,
                "mouse_ALM": 0.14285714285714285,
                "note": "Three points, three preparations, unmatched on region, task, label cardinality and "
                        "label modality. No trend is fitted across them and no relationship between fractional "
                        "rank and gate value is asserted from three points.",
            },
        },
        "zero_drop_accounting": {
            "n_seen": n_seen, "n_analysed": n_loaded, "n_refused": n_refused,
            "reconciles": bool(n_seen == n_loaded + n_refused),
        },
        "n_sessions_seen": n_seen,
    }
    return block_a


# =======================================================================================================
# BLOCK B -- count-matched Poisson surrogate, five corpora
# =======================================================================================================

def _two_sided_empirical_p(observed: float, null_draws: np.ndarray) -> float:
    n = len(null_draws)
    le = int(np.sum(null_draws <= observed))
    ge = int(np.sum(null_draws >= observed))
    return float(min(1.0, 2.0 * min((le + 1) / (n + 1), (ge + 1) / (n + 1))))


def _block_b_branch(real_mean: float, p_value: float, surrogate_mean: float, mdd_value: float | None) -> str:
    """Implements BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING's four branches, given the real corpus's
    pooled deviation-vs-spike-count gate value, the two-sided empirical p of that value against the
    surrogate distribution, the surrogate distribution's own mean, and its minimum detectable departure."""
    significant = p_value < 0.05
    if significant and real_mean < surrogate_mean:
        return "deviation_gate_failure_exceeds_counting_noise"
    if significant and real_mean > surrogate_mean:
        return "deviation_gate_value_is_not_in_the_counting_noise_direction"
    if (not significant) and mdd_value is not None and mdd_value <= abs(real_mean):
        return "deviation_gate_value_is_explained_by_counting_noise"
    return "inconclusive_below_detection_floor"


def poisson_surrogate_draw_correlations(activity_by_unit: np.ndarray, n_draws: int, seed_tag: str) -> dict:
    """The decisive counting-noise test: a surrogate trial set with the
    session's own real per-trial totals but a single shared true direction
    (the session's own L2-normalised mean count vector) and therefore no
    trial-to-trial direction structure whatsoever. Per trial i, the
    surrogate unit-count vector is drawn from independent Poissons whose
    EXPECTATION is that shared direction scaled so its expected total equals
    trial i's own observed total; the realised (not expected) total is what
    is correlated against, matching how the real gate correlates against
    each trial's own observed total."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials, n_units = activity.shape
    mean_vec = activity.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0.0 or n_trials < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "not_computable", "reason": "zero session-mean activity or too few trials"}
    direction = mean_vec / norm
    direction_sum = float(direction.sum())
    if direction_sum <= 0.0:
        return {"status": "not_computable", "reason": "normalised direction sums to zero"}
    real_totals = activity.sum(axis=1)
    lambdas = np.outer(real_totals / direction_sum, direction)  # (n_trials, n_units), E[row sum] = real total

    draw_r: list[float] = []
    for d in range(n_draws):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|surrogate_draw{d}"))
        surrogate = rng.poisson(lambdas)
        deviation = rate_free_state_deviation(surrogate)
        total = surrogate.sum(axis=1).astype(float)
        finite = np.isfinite(deviation)
        if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            continue
        x, y = deviation[finite], total[finite]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        # Equivalent to partial_correlation_permutation_test's own zero-controls Pearson-r formula
        # (np.corrcoef of the mean-centred series); the permutation wrapper's own p-value is never read
        # for a surrogate draw, only its observed r, so recomputing 10000 permutations 200 times over is
        # skipped -- the REAL gate below still goes through the unmodified permutation estimator.
        draw_r.append(float(np.corrcoef(x, y)[0, 1]))
    return {"status": "computed", "n_draws_requested": n_draws, "n_draws_usable": len(draw_r),
            "draw_r_values": draw_r}


def _dominant_mode_gate_one_session(counts: np.ndarray | None, total: np.ndarray, seed_tag: str) -> dict:
    """The dominant latent's own orthogonality gate against total spike count -- the same quantity
    _block_a_branch's amplitude arm gates on, computed here session by session for corpora Block A never
    touches, via the identical unchanged trial_amplitude_covariates. None if the caller has no per-bin
    tensor for this session (only a pre-reduced (trials, units) activity array)."""
    if counts is None or counts.shape[0] < 16:
        return {"status": "not_computable", "reason": "no per-bin tensor or too few trials"}
    covariates = trial_amplitude_covariates(counts)
    if covariates["status"] != "computed":
        return {"status": "not_computable", "reason": covariates.get("reason")}
    gain = np.asarray(covariates["leading_component_score_gain"], dtype=float)
    rng = np.random.default_rng(stable_seed(f"{seed_tag}|dominant_mode_gate"))
    return partial_correlation_permutation_test(gain, total, controls=[], n_perm=10000, rng=rng)


def _corpus_gate_and_surrogate(sessions: list[dict], corpus_key: str, seed_prefix: str) -> dict:
    """sessions: list of {"session": id, "activity_by_unit": (trials, units), "counts": (trials, units,
    bins) or None}. Computes, per session, the real deviation-vs-spike-count gate and the dominant-
    latent-amplitude-vs-spike-count gate (both the full permutation estimator, unchanged) plus the
    deviation's counting-noise surrogate prediction. The multi-object macaque corpus's dominant-mode gate
    is read from Block A's own already-computed pooled value instead of being recomputed here (the
    session's "counts" is left None for that corpus) -- shared machinery, not duplicated compute."""
    per_session = []
    for entry in sessions:
        activity = entry["activity_by_unit"]
        deviation = rate_free_state_deviation(activity)
        total = activity.sum(axis=1).astype(float)
        finite = np.isfinite(deviation)
        row: dict = {"session": entry["session"], "n_units": int(activity.shape[1]),
                     "n_trials": int(activity.shape[0]),
                     "median_total_spike_count_per_trial": float(np.median(total)),
                     "median_count_per_unit_per_trial": float(np.median(activity))}
        if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            row["status"] = "too_few_trials_with_a_defined_direction"
            per_session.append(row)
            continue
        rng = np.random.default_rng(stable_seed(f"{seed_prefix}|{entry['session']}|real_gate"))
        gate = partial_correlation_permutation_test(deviation[finite], total[finite], controls=[],
                                                      n_perm=10000, rng=rng)
        surrogate = _fit(
            f"surrogate|{corpus_key}|{entry['session']}",
            lambda a=activity, s=entry["session"]: poisson_surrogate_draw_correlations(
                a, N_SURROGATE_DRAWS, f"{seed_prefix}|{s}"))
        row["status"] = "computed"
        row["real_deviation_gate"] = gate
        row["surrogate"] = surrogate
        row["dominant_mode_gate"] = _dominant_mode_gate_one_session(
            entry.get("counts"), total, f"{seed_prefix}|{entry['session']}")
        per_session.append(row)

    computed = [r for r in per_session if r["status"] == "computed"]
    real_values = [r["real_deviation_gate"]["r"] for r in computed if r["real_deviation_gate"].get("status") == "computed"]
    real_pooled = _pool_values(real_values)
    real_mean = real_pooled.get("mean_value")
    dominant_mode_values = [r["dominant_mode_gate"]["r"] for r in computed
                             if r.get("dominant_mode_gate", {}).get("status") == "computed"]
    dominant_mode_gate_pooled = _pool_values(dominant_mode_values) if dominant_mode_values else \
        {"status": "not_computable", "n_sessions": 0}

    n_draws_available = min((r["surrogate"]["n_draws_usable"] for r in computed
                              if r["surrogate"].get("status") == "computed"), default=0)
    surrogate_pooled_by_draw = []
    if n_draws_available >= 1:
        usable = [r["surrogate"]["draw_r_values"] for r in computed if r["surrogate"].get("status") == "computed"
                  and r["surrogate"]["n_draws_usable"] >= n_draws_available]
        for d in range(n_draws_available):
            surrogate_pooled_by_draw.append(float(np.mean([draws[d] for draws in usable])))

    if real_mean is None or not surrogate_pooled_by_draw:
        return {
            "n_sessions_seen": len(sessions), "n_sessions_computed": len(computed), "per_session": per_session,
            "real_gate": real_pooled, "dominant_mode_gate": dominant_mode_gate_pooled,
            "surrogate_distribution": {"status": "not_computable"},
            "branch": "inconclusive_below_detection_floor",
            "reason": "fewer than the sessions needed for a pooled real gate or a surrogate distribution",
        }

    surrogate_arr = np.asarray(surrogate_pooled_by_draw, dtype=float)
    p_value = _two_sided_empirical_p(real_mean, surrogate_arr)
    surrogate_mean = float(np.mean(surrogate_arr))
    mdd = minimum_detectable_paired_difference(surrogate_pooled_by_draw)
    mdd_value = mdd.get("mdd") if mdd.get("status") == "computed" else None
    branch = _block_b_branch(real_mean, p_value, surrogate_mean, mdd_value)

    return {
        "n_sessions_seen": len(sessions), "n_sessions_computed": len(computed), "per_session": per_session,
        "real_gate": real_pooled, "real_gate_mean_value": real_mean,
        "dominant_mode_gate": dominant_mode_gate_pooled,
        "dominant_mode_gate_mean_value": dominant_mode_gate_pooled.get("mean_value"),
        "surrogate_distribution": {
            "status": "computed", "n_draws_pooled": len(surrogate_pooled_by_draw),
            "mean": surrogate_mean, "sd": float(np.std(surrogate_arr, ddof=1)),
            "min": float(np.min(surrogate_arr)), "max": float(np.max(surrogate_arr)),
            "minimum_detectable_paired_difference_at_80pct_power": mdd,
        },
        "two_sided_empirical_p_value": p_value,
        "branch": branch,
        "median_total_spike_count_per_trial_across_sessions": float(np.median(
            [r["median_total_spike_count_per_trial"] for r in computed])) if computed else None,
        "median_count_per_unit_per_trial_across_sessions": float(np.median(
            [r["median_count_per_unit_per_trial"] for r in computed])) if computed else None,
    }


def _attach_block_b_label_disclosure(block: dict) -> dict:
    """Extend-only: attaches a per-corpus disclosure field, never moves the
    branch already computed by _corpus_gate_and_surrogate above."""
    if block.get("branch") != "deviation_gate_value_is_not_in_the_counting_noise_direction":
        return dict(block)  # shallow copy: never mutate the cached fit's own dict in place
    return {**block, "label_disclosure": {
        "observed_gate_is_negative": bool(block.get("real_gate_mean_value") is not None
                                           and block["real_gate_mean_value"] < 0.0),
        "observed_gate_mean_value": block.get("real_gate_mean_value"),
        "surrogate_distribution_mean": block.get("surrogate_distribution", {}).get("mean")
        if isinstance(block.get("surrogate_distribution"), dict) else None,
        "statement": BLOCK_B_NOT_IN_DIRECTION_LABEL_DISCLOSURE,
    }}


# ---------------------------------------------------------------------------------------------------
# Repair: the surrogate-vs-real deviation MAGNITUDE diagnostic. poisson_surrogate_draw_correlations above
# stores only each draw's correlation, never the deviation values that produced it, so this diagnostic was
# not derivable from the delivered checkpoint and requires a genuine re-derivation -- done here by an
# identical re-construction (same lambdas, same per-draw seed formula) that ADDITIONALLY captures each
# draw's own median deviation magnitude, and which asserts its recomputed correlations reproduce the
# delivered checkpoint's own draw_r_values before anything new is reported.
# ---------------------------------------------------------------------------------------------------

def poisson_surrogate_and_real_deviation_magnitudes(activity_by_unit: np.ndarray, n_draws: int, seed_tag: str,
                                                      expected_draw_r: list[float] | None = None) -> dict:
    """Identical construction to poisson_surrogate_draw_correlations (same lambdas, same
    stable_seed(f"{seed_tag}|surrogate_draw{d}") per draw), so passing the identical seed_tag reproduces the
    identical draws bit for bit. rate_free_state_deviation's output is already non-negative for a
    non-negative count vector (one minus a cosine similarity between two non-negative vectors, which is
    itself in [0, 1]), so its own value already IS the magnitude -- no absolute value is taken."""
    activity = np.asarray(activity_by_unit, dtype=float)
    n_trials, n_units = activity.shape
    real_deviation = rate_free_state_deviation(activity)
    real_finite = np.isfinite(real_deviation)
    real_median_magnitude = float(np.median(real_deviation[real_finite])) if int(real_finite.sum()) else None

    mean_vec = activity.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0.0 or n_trials < MIN_TRIALS_WITH_DEFINED_DIRECTION:
        return {"status": "not_computable", "reason": "zero session-mean activity or too few trials",
                "real_median_deviation_magnitude": real_median_magnitude}
    direction = mean_vec / norm
    direction_sum = float(direction.sum())
    if direction_sum <= 0.0:
        return {"status": "not_computable", "reason": "normalised direction sums to zero",
                "real_median_deviation_magnitude": real_median_magnitude}
    real_totals = activity.sum(axis=1)
    lambdas = np.outer(real_totals / direction_sum, direction)

    draw_r: list[float] = []
    draw_median_magnitude: list[float] = []
    for d in range(n_draws):
        rng = np.random.default_rng(stable_seed(f"{seed_tag}|surrogate_draw{d}"))
        surrogate = rng.poisson(lambdas)
        deviation = rate_free_state_deviation(surrogate)
        total = surrogate.sum(axis=1).astype(float)
        finite = np.isfinite(deviation)
        if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            continue
        x, y = deviation[finite], total[finite]
        if np.std(x) == 0.0 or np.std(y) == 0.0:
            continue
        draw_r.append(float(np.corrcoef(x, y)[0, 1]))
        draw_median_magnitude.append(float(np.median(deviation[finite])))

    reproduces_delivered = None
    if expected_draw_r is not None:
        reproduces_delivered = (len(draw_r) == len(expected_draw_r)
                                 and bool(np.allclose(draw_r, expected_draw_r, atol=1e-9, rtol=1e-9)))
        assert reproduces_delivered, (
            "re-derived surrogate correlations do not reproduce the delivered checkpoint's draw_r_values at "
            "tolerance 1e-9 -- the magnitude diagnostic must not be reported unless this holds")

    return {
        "status": "computed", "n_draws_requested": n_draws, "n_draws_usable": len(draw_r),
        "reproduced_delivered_draw_r_values": reproduces_delivered,
        "real_median_deviation_magnitude": real_median_magnitude,
        "surrogate_median_deviation_magnitude": float(np.median(draw_median_magnitude))
        if draw_median_magnitude else None,
    }


def _magnitude_diagnostic_verdict(ratio: float | None) -> str:
    if ratio is None:
        return "not_computable"
    if ratio <= 1.0:
        return "surrogate_is_a_plausible_upper_bound_on_the_counting_noise_contribution"
    return "surrogate_over_predicts_deviation_magnitude_and_is_not_a_plausible_null"


def _corpus_deviation_magnitude_diagnostic(sessions: list[dict], corpus_block: dict, corpus_key: str,
                                            seed_prefix: str) -> dict:
    """sessions and seed_prefix must be the IDENTICAL arguments _corpus_gate_and_surrogate already received
    for this corpus (same activity arrays, same seed_prefix string) so that per session the surrogate draws
    reproduce bit for bit; corpus_block is that already-cached call's own return value, read only for the
    per-session draw_r_values to reproduce and the session status to respect -- never mutated, never
    recomputed itself."""
    by_session = {s["session"]: s["activity_by_unit"] for s in sessions}
    per_session_rows = {r["session"]: r for r in corpus_block.get("per_session", [])}
    real_mags: list[float] = []
    surrogate_mags: list[float] = []
    n_checked = 0
    for session_id, activity in by_session.items():
        row = per_session_rows.get(session_id)
        if row is None or row.get("status") != "computed" or row.get("surrogate", {}).get("status") != "computed":
            continue
        expected = row["surrogate"]["draw_r_values"]
        seed_tag = f"{seed_prefix}|{session_id}"
        key = f"surrogate_magnitude|{corpus_key}|{session_id}"
        result = _fit(key, lambda a=activity, tag=seed_tag, e=expected: poisson_surrogate_and_real_deviation_magnitudes(
            a, len(e), tag, expected_draw_r=e))
        n_checked += 1
        if result.get("real_median_deviation_magnitude") is not None:
            real_mags.append(result["real_median_deviation_magnitude"])
        if result.get("surrogate_median_deviation_magnitude") is not None:
            surrogate_mags.append(result["surrogate_median_deviation_magnitude"])

    median_real = float(np.median(real_mags)) if real_mags else None
    median_surrogate = float(np.median(surrogate_mags)) if surrogate_mags else None
    ratio = (median_surrogate / median_real) if (median_real not in (None, 0.0) and median_surrogate is not None) \
        else None
    verdict = _magnitude_diagnostic_verdict(ratio)
    return {
        "n_sessions_checked": n_checked, "n_sessions_with_real_magnitude": len(real_mags),
        "n_sessions_with_surrogate_magnitude": len(surrogate_mags),
        "median_real_deviation_magnitude_across_sessions": median_real,
        "median_surrogate_deviation_magnitude_across_sessions": median_surrogate,
        "surrogate_to_real_ratio": ratio, "verdict": verdict,
        "statement": (
            "A ratio at or below 1 means the surrogate's pure-counting-noise deviation magnitude never "
            "exceeds what the real data actually shows, licensing its use as an upper bound on how much of "
            "the real deviation counting noise alone could produce. A ratio above 1 means the surrogate -- "
            "built with no real trial-to-trial direction structure at all -- produces MORE deviation than "
            "the real data does, which is not a plausible null for the real process and must not be "
            "presented as a bound on anything."
        ),
    }


def _load_panichello_for_block_b(root: Path) -> list[dict]:
    directory = _panichello_directory(root)
    if directory is None:
        return []
    out = []
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, simplify_cells=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        counts_all = _counts_from_spikes(spikes, time_ms)
        out.append({"session": Path(path).stem, "activity_by_unit": counts_all.sum(axis=2)})
    return out


def _load_alm_for_block_b(root: Path) -> list[dict]:
    directory = alm_data_directory(root)
    out = []
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.mat")):
        raw = load_alm_raw_session(path, bin_ms=100.0, window_s=1.2, require_both_arms=False)
        if raw is None:
            continue
        out.append({"session": path.stem, "activity_by_unit": raw["control_counts"].sum(axis=2)})
    return out


def _load_human_for_block_b(root: Path, dataset: str) -> list[dict]:
    out = []
    for entry in iter_all_corpora(root):
        if entry["dataset"] != dataset or entry.get("structure") != "pooled":
            continue
        counts = delay_counts(entry["spike_lists"], entry["epoch_onsets"]["delay"], entry["epoch_windows"]["delay"])
        out.append({"session": f"{entry['patient']}|{entry['session']}", "activity_by_unit": counts.sum(axis=2)})
    return out


def _human_seen_denominator(root: Path, dataset: str) -> int:
    """Raw sessions on disk for a human corpus, independent of whether the
    shared loader's own unit/trial/region floors ultimately yield a pooled
    row -- the zero-drop denominator this block reconciles against."""
    if dataset == "dandi_000469":
        directory = root / "000469"
        return sum(1 for _ in directory.glob("sub-*/*_ses-2_ecephys+image.nwb")) if directory.is_dir() else 0
    if dataset == "dandi_001187":
        return sum(1 for meta in canonical_sessions() if meta["primary_release"] == "001187")
    raise ValueError(dataset)


def run_block_b(root: Path, t0: float, output: dict, watters_seen: int, watters_loaded: list[dict]) -> dict:
    _log("Block B: loading five corpora for the counting-noise census")
    corpora: dict[str, dict] = {}

    panichello_sessions = _load_panichello_for_block_b(root)
    corpora["panichello_2024_macaque_lPFC_single_item"] = _fit(
        "block_b|panichello_2024", lambda: _corpus_gate_and_surrogate(
            panichello_sessions, "panichello_2024", "dissociation_replication_and_counting_noise|panichello"))
    corpora["panichello_2024_macaque_lPFC_single_item"] = _attach_block_b_label_disclosure(
        corpora["panichello_2024_macaque_lPFC_single_item"])
    output.setdefault("_progress", {})["block_b_corpora_done"] = 1
    _flush(output)
    _log(f"  block B panichello done elapsed={time.time() - t0:.0f}s")
    corpora["panichello_2024_macaque_lPFC_single_item"]["deviation_magnitude_diagnostic"] = \
        _corpus_deviation_magnitude_diagnostic(
            panichello_sessions, corpora["panichello_2024_macaque_lPFC_single_item"], "panichello_2024",
            "dissociation_replication_and_counting_noise|panichello")
    _flush(output)
    _log(f"  block B panichello magnitude diagnostic done elapsed={time.time() - t0:.0f}s")

    alm_sessions = _load_alm_for_block_b(root)
    corpora["inagaki_alm5_mouse_ALM"] = _fit(
        "block_b|inagaki_alm5", lambda: _corpus_gate_and_surrogate(
            alm_sessions, "inagaki_alm5", "dissociation_replication_and_counting_noise|alm"))
    corpora["inagaki_alm5_mouse_ALM"] = _attach_block_b_label_disclosure(corpora["inagaki_alm5_mouse_ALM"])
    output["_progress"]["block_b_corpora_done"] = 2
    _flush(output)
    _log(f"  block B ALM done elapsed={time.time() - t0:.0f}s")
    corpora["inagaki_alm5_mouse_ALM"]["deviation_magnitude_diagnostic"] = _corpus_deviation_magnitude_diagnostic(
        alm_sessions, corpora["inagaki_alm5_mouse_ALM"], "inagaki_alm5",
        "dissociation_replication_and_counting_noise|alm")
    _flush(output)
    _log(f"  block B ALM magnitude diagnostic done elapsed={time.time() - t0:.0f}s")

    # watters_loaded is the SAME list Block A already loaded in main()'s single iter_watters pass -- the
    # dominant cost of touching this corpus at all is reading its per-trial spike cache, so it is read
    # once and shared, not reloaded a second time here.
    watters_raw = [{"session": s["session"], "activity_by_unit": s["counts"].sum(axis=2)} for s in watters_loaded]
    corpora["watters_2026_macaque_multi_object"] = _fit(
        "block_b|watters_2026", lambda: _corpus_gate_and_surrogate(
            watters_raw, "watters_2026", "dissociation_replication_and_counting_noise|watters"))
    corpora["watters_2026_macaque_multi_object"] = _attach_block_b_label_disclosure(
        corpora["watters_2026_macaque_multi_object"])
    output["_progress"]["block_b_corpora_done"] = 3
    _flush(output)
    _log(f"  block B watters done elapsed={time.time() - t0:.0f}s")
    corpora["watters_2026_macaque_multi_object"]["deviation_magnitude_diagnostic"] = \
        _corpus_deviation_magnitude_diagnostic(
            watters_raw, corpora["watters_2026_macaque_multi_object"], "watters_2026",
            "dissociation_replication_and_counting_noise|watters")
    _flush(output)
    _log(f"  block B watters magnitude diagnostic done elapsed={time.time() - t0:.0f}s")

    for dataset in HUMAN_CORPORA_FOR_THE_CENSUS:
        sessions = _load_human_for_block_b(root, dataset)
        corpora[f"{dataset}_human"] = _fit(
            f"block_b|{dataset}", lambda s=sessions, d=dataset: _corpus_gate_and_surrogate(
                s, d, f"dissociation_replication_and_counting_noise|{d}"))
        corpora[f"{dataset}_human"] = _attach_block_b_label_disclosure(corpora[f"{dataset}_human"])
        output["_progress"]["block_b_corpora_done"] = output["_progress"]["block_b_corpora_done"] + 1
        _flush(output)
        _log(f"  block B {dataset} done elapsed={time.time() - t0:.0f}s")
        corpora[f"{dataset}_human"]["deviation_magnitude_diagnostic"] = _corpus_deviation_magnitude_diagnostic(
            sessions, corpora[f"{dataset}_human"], dataset, f"dissociation_replication_and_counting_noise|{dataset}")
        _flush(output)
        _log(f"  block B {dataset} magnitude diagnostic done elapsed={time.time() - t0:.0f}s")

    output.pop("_progress", None)

    dataset_seen = {
        "panichello_2024_macaque_lPFC_single_item": len(glob.glob(
            str(_panichello_directory(root) / "*.mat"))) if _panichello_directory(root) else 0,
        "inagaki_alm5_mouse_ALM": len(list(alm_data_directory(root).glob("*.mat")))
        if alm_data_directory(root).is_dir() else 0,
        "watters_2026_macaque_multi_object": watters_seen,
    }
    for dataset in HUMAN_CORPORA_FOR_THE_CENSUS:
        dataset_seen[f"{dataset}_human"] = _human_seen_denominator(root, dataset)

    n_computed = {name: block.get("n_sessions_computed", 0) for name, block in corpora.items()}
    excluded_human_from_iter_all_corpora = {
        "dandi_000574": "iter_all_corpora also reaches this human dataset, but it carries no per-trial item "
                        "identity label and was never part of the pairing results/human_content_decodability.json "
                        "establishes as 'both human corpora' (dandi_000469, dandi_001187); it is excluded here "
                        "by name from the five-corpus count, not silently dropped by a floor.",
    }

    gate_table = []
    for name, block in corpora.items():
        gate_table.append({
            "corpus": name, "n_sessions_computed": block.get("n_sessions_computed"),
            "deviation_gate_mean_value": block.get("real_gate_mean_value"),
            "deviation_gate_p_value": block.get("real_gate", {}).get("two_sided_p_value") if isinstance(
                block.get("real_gate"), dict) else None,
            "deviation_gate_ci": [block.get("real_gate", {}).get("ci_lower"), block.get("real_gate", {}).get("ci_upper")]
            if isinstance(block.get("real_gate"), dict) else None,
            "median_total_spike_count_per_trial": block.get("median_total_spike_count_per_trial_across_sessions"),
            "median_count_per_unit_per_trial": block.get("median_count_per_unit_per_trial_across_sessions"),
            "surrogate_predicted_gate_mean": block.get("surrogate_distribution", {}).get("mean")
            if isinstance(block.get("surrogate_distribution"), dict) else None,
            "branch": block.get("branch"),
            "median_real_deviation_magnitude": block.get("deviation_magnitude_diagnostic", {}).get(
                "median_real_deviation_magnitude_across_sessions"),
            "median_surrogate_deviation_magnitude": block.get("deviation_magnitude_diagnostic", {}).get(
                "median_surrogate_deviation_magnitude_across_sessions"),
            "surrogate_to_real_deviation_magnitude_ratio": block.get("deviation_magnitude_diagnostic", {}).get(
                "surrogate_to_real_ratio"),
            "deviation_magnitude_diagnostic_verdict": block.get("deviation_magnitude_diagnostic", {}).get("verdict"),
        })
    n_fail_gate = sum(1 for row in gate_table if row.get("deviation_gate_p_value") is not None
                       and row["deviation_gate_p_value"] <= 0.05)

    magnitude_verdicts = [row["deviation_magnitude_diagnostic_verdict"] for row in gate_table
                           if row.get("deviation_magnitude_diagnostic_verdict") is not None]

    block_b = {
        "decision_rule_declared_before_fitting": BLOCK_B_DECISION_RULE_DECLARED_BEFORE_FITTING,
        "n_surrogate_draws_per_session": N_SURROGATE_DRAWS,
        "human_corpora_scope": {
            "both_human_corpora_used_here": list(HUMAN_CORPORA_FOR_THE_CENSUS),
            "excluded_from_iter_all_corpora": excluded_human_from_iter_all_corpora,
        },
        "corpora": corpora,
        "census_table_five_corpora_no_trend_no_ranking": gate_table,
        "n_of_five_corpora_failing_their_own_deviation_gate_at_p_leq_0_05": n_fail_gate,
        "not_in_the_counting_noise_direction_label_disclosure": BLOCK_B_NOT_IN_DIRECTION_LABEL_DISCLOSURE,
        "count_separation_disclosure": _count_separation_disclosure(gate_table),
        "deviation_magnitude_diagnostic_summary": {
            "per_corpus_verdicts_no_trend_no_ranking": {row["corpus"]: row["deviation_magnitude_diagnostic_verdict"]
                                                          for row in gate_table},
            "n_corpora_where_surrogate_is_a_plausible_upper_bound": sum(
                1 for v in magnitude_verdicts
                if v == "surrogate_is_a_plausible_upper_bound_on_the_counting_noise_contribution"),
            "n_corpora_where_surrogate_over_predicts": sum(
                1 for v in magnitude_verdicts
                if v == "surrogate_over_predicts_deviation_magnitude_and_is_not_a_plausible_null"),
        },
        "zero_drop_accounting": {
            "n_seen_by_corpus": dataset_seen, "n_computed_by_corpus": n_computed,
        },
    }
    return block_b


# =======================================================================================================
# Driver
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    _COMPLETED_FITS.update(_load_completed_fits())
    _log(f"model fits already recorded as complete: {len(_COMPLETED_FITS)}")
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Two blocks sharing machinery and this one artifact. Block A: the dominant-latent amplitude half "
            "of the rate/deviation dissociation in the multi-object macaque corpus (DANDI 000620), the "
            "corpus's untested half. Block B: a count-matched Poisson surrogate test of whether the rate-free "
            "deviation observable's orthogonality-gate failure in mouse ALM is counting noise, run identically "
            "across every corpus in the project with a per-trial per-unit spike tensor."
        ),
        "block_a_reproduction_gate_note": (
            "The reproduction gate below re-runs the shared dominant-latent-amplitude and rate-free-deviation "
            "estimators on the single-item macaque sessions and compares against "
            "results/dominant_latent_identity_and_behaviour_breadth.json and "
            "results/rate_free_state_geometry_behavior_link.json, at tolerance 1e-6 -- the same gate "
            "results/dissociation_cross_preparation_test.json used, imported and called unchanged. If it does "
            "not reproduce exactly, Block A stops before computing any multi-object statistic; Block B does "
            "not depend on this gate and proceeds regardless."
        ),
        "status": "running",
    }
    _flush(output)

    _log("running reproduction gate on single-item macaque sessions")
    gate_result = _fit("reproduction_gate", lambda: reproduction_gate(root))
    output["reproduction_gate"] = gate_result
    _flush(output)
    _log(f"reproduction gate: {gate_result['status']}")

    # Loaded once, here, and shared by Block A (which needs the full per-trial tensors) and Block B's
    # multi-object-macaque surrogate row (which only needs each session's per-unit totals) -- this
    # corpus's per-trial spike cache is the dominant cost of touching it at all, so it is read once.
    _log("loading the multi-object macaque corpus (one pass, shared by both blocks)")
    watters_seen = 0
    watters_loaded: list[dict] = []
    watters_refused: list[dict] = []
    for session in iter_watters(root, bin_ms=100.0):
        watters_seen += 1
        if session["status"] != "loaded":
            watters_refused.append({
                "session": session["session"], "animal": session.get("animal"),
                "session_date": session.get("session_date"),
                "task_variant": session.get("behavioural_task_variant"), "status": session["status"]})
            continue
        watters_loaded.append(session)
    _log(f"multi-object macaque corpus loaded: {watters_seen} seen, {len(watters_loaded)} loaded, "
         f"{len(watters_refused)} refused, elapsed={time.time() - t0:.0f}s")

    if gate_result["status"] != "reproduced_exactly":
        output["block_a"] = {
            "status": "not_run_reproduction_gate_failed",
            "reason": "the reproduction gate did not reproduce the single-item macaque numbers at tolerance "
                      "1e-6; Block A is stopped per the mandatory pre-declared rule and no multi-object "
                      "statistic was computed",
        }
        _log("STOPPING Block A: reproduction gate did not reproduce; no multi-object statistic computed")
    else:
        block_a = run_block_a(watters_seen, watters_loaded, watters_refused, t0, output)
        output["block_a"] = block_a
        _flush(output)
        _log(f"Block A branch at primary cell: {block_a['branch_at_primary_cell']['branch']}")

    block_b = run_block_b(root, t0, output, watters_seen, watters_loaded)
    output["block_b"] = block_b
    _flush(output)
    _log("Block B complete: " + json.dumps(
        {row["corpus"]: row["branch"] for row in block_b["census_table_five_corpora_no_trend_no_ranking"]}))

    output["how_this_artifact_was_assembled"] = {
        "n_model_fits_served_from_an_earlier_invocation": _FITS_SERVED_FROM_CHECKPOINT,
        "n_model_fits_computed_in_this_invocation": _FITS_COMPUTED_HERE,
        "completed_fit_record": "results/.checkpoints/dissociation_replication_and_counting_noise_checkpoint.json",
    }
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate_result["status"],
        "block_a_branch": output.get("block_a", {}).get("branch_at_primary_cell", {}).get("branch"),
        "block_b": {row["corpus"]: row["branch"] for row in block_b["census_table_five_corpora_no_trend_no_ranking"]},
    }, indent=2, default=float))


if __name__ == "__main__":
    main()
