"""run_between_session_component_behaviour_state.py -- is the accuracy-
predicting rate-free direction-deviation component's behavioural signal a
slow state that varies BETWEEN recording sessions (between participants, in
humans), rather than a trial-by-trial fluctuation?

Four delivered candidate positive identities for this component were voided
by the same control: replacing every trial's value with its own session's
mean and recomputing reproduced each result in sign and significance.
Arithmetically, that control IS the between-session association. A void of a
trial-level claim is not a demonstration that nothing is there -- it is a
demonstration that whatever is there may not live at the trial level. This
module asks the between-session question directly, in every staged corpus
that carries both the component and a behavioural outcome, animal and human
alike. It does not amend, soften or reinterpret any voided trial-level
branch; those stand exactly as fired.

THE THREAT. A between-session correlation is confounded by everything that
varies between sessions: how many units were isolated, how many trials were
run, how much total spike count a session carries, and how far into a
recording series the session sits. Unit yield is the sharpest -- the
component is a direction statistic in a space whose dimension is the unit
count, so a session-level correlation between it and accuracy is the
EXPECTED nuisance result. The primary statistic here is therefore NOT the
raw between-session correlation but the between-session correlation after
partialling four session-level nuisances: isolated unit count, trial count,
mean total spike count, and order within the recording series. The raw
number is always reported beside the partialled one and never alone. If the
association does not survive that partial, that is reported AS the answer
and no further parameterisation is tried.

MULTI-OBJECT CORPUS. The multi-object macaque corpus is analysed WITHIN
item-count level throughout and combined across levels by a
trial-count-weighted average of the within-level estimates (effect size)
with an inverse-variance Fisher-z combination of the within-level tests
(inference). A pooled-across-item-count number is never this corpus's
effect size; that estimator is known to reverse sign relative to the
within-level one in this corpus, so the generic estimator refuses
mixed-item-count input outright.

HUMANS. In the human corpora the clustering unit is the participant, not
the session. The delivered human trial-level maintenance result is a
powered null and is never restated here as humans lacking the component; a
powered trial-level null is entirely compatible with a between-participant
association, which is what this module measures. Human associations are
computed within set-size level (load determines both task difficulty and
the sampling of error trials) and combined across levels like the
multi-object corpus.

POWER. Before any result is read, every cell declares the minimum
detectable correlation at 80 percent power for its own number of clustering
units, against this project's standing behavioural reference of 0.14 r
units. A non-significant cell whose minimum detectable correlation exceeds
that reference is reported below its detection floor -- never as agreement,
never as disagreement, and never without both numbers in the same field.

WITHIN VERSUS BETWEEN. The within-session (within-participant, in humans)
statistic is reported beside the between-session statistics as a separate
number and is never pooled with them. If the two disagree in sign, both are
reported and the within-session one is named the commensurable one.

Development aid: setting the environment variable
BETWEEN_SESSION_STATE_SCOPE_LIMIT limits every corpus to its first N
clustering units -- staging stops enumerating once N clustering units have
been seen, aggregate statistics are computed only on those units and their
checkpoint keys are qualified with the limit so a full run can never serve
them, and the reproduction gate is reported but does not stop the run
(it cannot reproduce delivered numbers from a truncated corpus). Every one
of these effects is recorded in the artifact's scope block whenever the
limit is active. A full run leaves it unset.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import data_root, iter_watters  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_dissociation_cross_preparation_test import MIN_TRIALS_WITH_DEFINED_DIRECTION  # noqa: E402
from run_dissociation_replication_and_counting_noise import (  # noqa: E402
    _observable_arrays, _session_observable_arm,
)
from run_dominant_latent_identity_and_behaviour_breadth import (  # noqa: E402
    _load_session as _macaque_load_session, _session_paths as _macaque_session_paths,
)
from run_human_maintenance_behaviour_link import ADMISSION_ITERATORS  # noqa: E402
from run_rate_free_state_geometry_behavior_link import rate_free_state_deviation  # noqa: E402
from run_state_behavior_link import (  # noqa: E402
    MIN_ERROR_TRIALS_FOR_REACHABILITY, trial_amplitude_covariates,
)
from run_state_content_link import delay_counts  # noqa: E402
from run_watters_state_geometry import (  # noqa: E402
    MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION, PRIMARY_QUALITY_TIER, _pool_values,
)
from scipy.stats import norm  # noqa: E402
from state_persistence import slope_across_sessions_test  # noqa: E402
from statistics import (  # noqa: E402
    fdr_bh, forest_meta, minimum_detectable_paired_difference, partial_correlation_permutation_test,
    pearson_permutation_test, stable_seed,
)

OUTPUT_PATH = ROOT / "results" / "between_session_component_behaviour_state.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_between_session_component_behaviour_state"
ANALYSIS_VERSION = "2026-08-26"

MODULE_TAG = "run_between_session_component_behaviour_state"
BIN_MS = 100.0
N_PERM_PRIMARY = 10000   # between-unit tests feed headline decisions
N_PERM_PATIENT_WITHIN = 2000   # per-participant within-statistic p-values are diagnostic only
MIN_TRIALS_PER_PARTICIPANT_WITHIN = 16
ALPHA = 0.05
POWER_TARGET = 0.80
REFERENCE_R_UNITS = 0.14
MIN_CLUSTERING_UNITS_FOR_PARTIAL = 8   # four covariates: fewer units cannot support the declared partial
REPRODUCTION_TOLERANCE = 1e-6

SCOPE_LIMIT = os.environ.get("BETWEEN_SESSION_STATE_SCOPE_LIMIT")


def _k(key: str) -> str:
    """Aggregate checkpoint keys are qualified with any active scope limit so a truncated run can
    never serve its statistics to a full run; per-unit fit keys are unaffected because one unit's
    fit does not depend on which other units were staged."""
    return f"{key}|limit{SCOPE_LIMIT}" if SCOPE_LIMIT else key


def _limit_reached(n_seen: int) -> bool:
    return bool(SCOPE_LIMIT and n_seen >= int(SCOPE_LIMIT))

# Read-only reuse of completed per-session fits written by an earlier invocation of the delivered
# multi-object analysis. This module never writes those files; anything they lack is computed here and
# stored under this module's own checkpoint directory instead.
WATTERS_ARM_CHECKPOINT = ROOT / "results" / ".checkpoints" / "component_effect_size_and_anatomy_checkpoint.json"

REPRODUCTION_SOURCE_ARTIFACT = ROOT / "results" / "component_effect_size_and_anatomy.json"
BIAS_ONLY_CONTROL_SOURCE_ARTIFACT = ROOT / "results" / "swap_versus_imprecision_by_item_count.json"

PARTIALLED_COVARIATES = (
    "clustering_unit_isolated_unit_count",
    "clustering_unit_trial_count",
    "clustering_unit_mean_total_spike_count",
    "order_within_the_recording_series",
)

PRIMARY_DECISION_RULE_DECLARED_BEFORE_READING_RESULTS = (
    "Per corpus the clustering unit is the recording session for animal corpora and the participant for "
    "human corpora. One record per clustering unit carries: the mean component value over that unit's "
    "admitted trials (for the multi-object corpus and the human corpora, within one item-count or "
    "set-size level), the mean worse-behaviour value over the same trials, and four nuisance "
    "covariates: isolated unit count, trial count, mean total spike count, and order within the "
    "recording series.\n"
    "RAW family: the between-unit Pearson correlation of component mean versus behaviour mean.\n"
    "PARTIALLED family: the same correlation after residualising both variables on [intercept, unit "
    "count, trial count, mean total spike count, series order], by the residual method with an "
    "outcome-shuffling permutation null.\n"
    "Level-structured corpora (multi-object by item count, humans by set size): every association is "
    "computed WITHIN one level and combined across levels two ways, both reported -- the primary "
    "EFFECT SIZE is the trial-count-weighted average of within-level r values, and the primary "
    "INFERENCE is the inverse-variance random-effects combination (forest_meta) of within-level "
    "Fisher-z values with standard error 1/sqrt(n_level - 3), back-transformed to r. Raw trials are "
    "never pooled across levels.\n"
    "Multiplicity: Benjamini-Hochberg applied separately within the RAW family and within the "
    "PARTIALLED family, across the computable corpora plus the pooled row. Significance means "
    "q <= 0.05; uncorrected p is reported beside every q.\n"
    "Power floors, fixed before reading results: the minimum detectable correlation at 80 percent "
    "power for n independent clustering units is tanh((z_(1-alpha/2) + z_power)/sqrt(n - 3)) at "
    "alpha = 0.05 two-sided, power = 0.80; fewer than 5 units cannot support the formula, and such a "
    "cell is below its floor by construction. Every cell reports its floor regardless of outcome. A "
    "NON-SIGNIFICANT cell whose floor exceeds the standing 0.14 reference carries the label "
    "inconclusive_below_detection_floor with both numbers in one field -- never as agreement, never as "
    "disagreement.\n"
    "Corpus branches, checked in this fixed order:\n"
    "  1. partialled q <= 0.05 -> 'the_components_behavioural_signal_is_a_slow_between_session_state'.\n"
    "  2. raw q <= 0.05 and partialled q > 0.05 -> "
    "'the_between_session_association_is_explained_by_session_level_recording_nuisances'.\n"
    "  3. neither significant and the cell's minimum detectable correlation > 0.14 -> "
    "'inconclusive_below_detection_floor', carrying the cell's own floor and the reference in the same "
    "field.\n"
    "  4. otherwise -> 'no_between_session_association_at_either_level' (a powered null at the "
    "reference).\n"
    "Pooled verdict over all corpora, checked in this fixed order:\n"
    "  1. pooled PARTIALLED q <= 0.05 and every corpus with a computable PARTIALLED estimate shares the "
    "pooled sign -> 'the_components_behavioural_signal_is_a_slow_between_session_state'.\n"
    "  2. pooled RAW q <= 0.05 and pooled PARTIALLED q > 0.05 -> "
    "'the_between_session_association_is_explained_by_session_level_recording_nuisances'.\n"
    "  3. at least one non-human corpus significant in either family at its own q <= 0.05 and no human "
    "corpus significant in either family -> "
    "'the_between_session_association_is_present_in_non_human_corpora_only', with every human corpus's "
    "minimum detectable correlation stated alongside so a reader can tell a species difference from a "
    "power difference.\n"
    "  4. no corpus significant in either family and every corpus below its floor -> "
    "'inconclusive_below_detection_floor', carrying each corpus's floor and the reference in the same "
    "field.\n"
    "  5. otherwise -> 'no_between_session_association_at_either_level'.\n"
    "No branch fired here amends any previously fired trial-level branch: those remain exactly as "
    "fired, and a positive answer coexists with them."
)

WITHIN_VS_BETWEEN_DECLARATION = (
    "Within-session statistics live under 'within_session_statistic'; between-session statistics live "
    "under 'between_session_statistics'. No field of this artifact contains a statistic pooled across "
    "those two, and no branch reads one as the other. When they disagree in sign, both are reported "
    "and the within-session one is named the commensurable one."
)

MULTI_OBJECT_ESTIMATOR_DECLARATION = (
    "This corpus is analysed WITHIN item-count level throughout. Every between-session quantity is "
    "computed on one item-count level's trials at a time and combined across levels by a "
    "trial-count-weighted average of within-level estimates (effect size) and an inverse-variance "
    "Fisher-z combination of within-level tests (inference). A pooled-across-item-count estimator is "
    "never computed: the generic estimator refuses mixed-item-count records outright."
)


# =======================================================================================================
# Checkpointing (this module's own directory only; temp file + os.replace; completion flag written only
# after the compute returns)
# =======================================================================================================

def _checkpoint_path(key: str) -> Path:
    safe = key.replace("/", "_").replace("|", "__").replace(" ", "_")
    return CHECKPOINT_DIR / f"{safe}.json"


def _fit(key: str, compute):
    path = _checkpoint_path(key)
    try:
        record = json.loads(path.read_text())
    except (OSError, ValueError):
        record = None
    if isinstance(record, dict) and record.get("complete") is True:
        return record["value"]
    value = compute()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe({"complete": True, "value": value}), allow_nan=False, default=float))
    os.replace(scratch, path)
    return value


def _read_foreign_fit(path: Path, key: str):
    """Read-only lookup of a completed fit in another invocation's checkpoint file; never writes."""
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    entry = entries.get(key) if isinstance(entries, dict) else None
    if isinstance(entry, dict) and entry.get("complete") is True:
        return entry["value"]
    return None


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    scratch = OUTPUT_PATH.with_suffix(".partial")
    scratch.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    os.replace(scratch, OUTPUT_PATH)


# =======================================================================================================
# Power floors (fixed before any result is read)
# =======================================================================================================

def minimum_detectable_correlation(n_units: int, alpha: float = ALPHA, power: float = POWER_TARGET) -> dict:
    """Smallest true Pearson correlation a two-sided test at ``alpha`` had ``power`` to detect with
    ``n_units`` independent clustering units, on the Fisher-z scale."""
    if n_units < 5:
        return {"status": "not_computable", "n": int(n_units),
                "reason": "fewer than 5 clustering units cannot support the Fisher-z power formula",
                "alpha": alpha, "power": power}
    z_crit = float(norm.ppf(1.0 - alpha / 2.0))
    z_pow = float(norm.ppf(power))
    mdd = float(np.tanh((z_crit + z_pow) / np.sqrt(n_units - 3)))
    return {"status": "computed", "n": int(n_units), "alpha": alpha, "power": power,
            "minimum_detectable_correlation": mdd,
            "standing_reference_r_units": REFERENCE_R_UNITS,
            "exceeds_standing_reference": bool(mdd > REFERENCE_R_UNITS)}


def _fisher_z(r: float) -> float:
    return float(np.arctanh(np.clip(r, -1 + 1e-9, 1 - 1e-9)))


def _fisher_back(z: float) -> float:
    return float(np.tanh(z))


def _sign(value) -> int | None:
    if value is None or not np.isfinite(value):
        return None
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


# =======================================================================================================
# Core estimators (pure functions; the test file drives these on planted data)
# =======================================================================================================

class MixedItemCountsError(ValueError):
    """Raised when the multi-object corpus's records reach the generic estimator without level
    stratification."""


def forbid_mixed_item_count_records(records: list[dict]) -> None:
    """Every multi-object record declares exactly one item count; mixed-level input raises rather than
    pooling, because pooling across item count reverses this corpus's sign."""
    levels = {r.get("item_count") for r in records}
    if len(levels) > 1:
        raise MixedItemCountsError(
            "records span more than one item count; this corpus is analysed within item-count level only")


def build_unit_record(component_values: np.ndarray, behaviour_values: np.ndarray, unit_count: int,
                      series_order: int, mean_total_spike_count: float,
                      item_count: int | None = None) -> dict:
    component_values = np.asarray(component_values, dtype=float)
    behaviour_values = np.asarray(behaviour_values, dtype=float)
    finite = np.isfinite(component_values) & np.isfinite(behaviour_values)
    return {
        "component_mean": float(np.mean(component_values[finite])),
        "behaviour_mean": float(np.mean(behaviour_values[finite])),
        "n_trials": int(finite.sum()),
        "unit_count": int(unit_count),
        "mean_total_spike_count_per_trial": float(mean_total_spike_count),
        "series_order": int(series_order),
        "item_count": None if item_count is None else int(item_count),
    }


def between_session_stats(records: list[dict], seed_tag: str, n_perm: int = N_PERM_PRIMARY) -> dict:
    """Raw and nuisance-partialled between-unit association for ONE homogeneous clustering stratum
    (one corpus, or one item-count / set-size level of one corpus), plus the stratum's power floor."""
    forbid_mixed_item_count_records(records)
    n = len(records)
    floor = minimum_detectable_correlation(n)
    base = {
        "n_clustering_units": n,
        "covariates_partialled": list(PARTIALLED_COVARIATES),
        "minimum_detectable_correlation_at_80pct_power": floor,
    }
    if n < MIN_CLUSTERING_UNITS_FOR_PARTIAL:
        return {**base, "status": "too_few_clustering_units_for_the_declared_partial",
                "floor_requires_at_least_this_many_units": MIN_CLUSTERING_UNITS_FOR_PARTIAL}
    component = np.array([r["component_mean"] for r in records], dtype=float)
    behaviour = np.array([r["behaviour_mean"] for r in records], dtype=float)
    controls = [
        np.array([float(r["unit_count"]) for r in records]),
        np.array([float(r["n_trials"]) for r in records]),
        np.array([float(r["mean_total_spike_count_per_trial"]) for r in records]),
        np.array([float(r["series_order"]) for r in records]),
    ]
    rng_raw = np.random.default_rng(stable_seed(f"{seed_tag}|raw"))
    raw = pearson_permutation_test(component, behaviour, n_perm=n_perm, rng=rng_raw)
    if not np.isfinite(raw["r"]):
        return {**base, "status": "not_computable_zero_variance_in_either_variable",
                "reason": "the component mean or the behaviour mean is constant across clustering units"}
    rng_part = np.random.default_rng(stable_seed(f"{seed_tag}|partialled"))
    part = partial_correlation_permutation_test(behaviour, component, controls, n_perm=n_perm, rng=rng_part)
    if part.get("status") != "computed" or not np.isfinite(part.get("r", float("nan"))):
        return {**base, "status": "partialled_not_computable",
                "reason": "residualising on the nuisance covariates left a zero-variance residual",
                "raw": {"status": "computed", "r": raw["r"], "p_value": raw["p_value"],
                         "p_analytic": raw["p_analytic"], "n": raw["n"]}}
    out = {
        **base,
        "status": "computed",
        "raw": {"status": "computed", "r": raw["r"], "p_value": raw["p_value"],
                 "p_analytic": raw["p_analytic"], "n": raw["n"]},
        "partialled_after_recording_nuisances": {
            "status": "computed", "r": part["r"], "p_value": part["p_value"],
            "p_analytic": part["p_analytic"], "n": part["n"], "n_controls": part["n_controls"]},
    }
    for key in ("raw", "partialled_after_recording_nuisances"):
        entry = out[key]
        entry["below_detection_floor"] = bool(
            floor.get("status") == "computed" and floor["minimum_detectable_correlation"] > REFERENCE_R_UNITS)
        entry["standing_reference_r_units"] = REFERENCE_R_UNITS
    return out


def combine_levels(level_stats: list[dict], level_weights: list[int], labels: list[str]) -> dict:
    """Combine WITHIN-LEVEL between-unit estimates of one level-structured corpus.

    Primary effect size: trial-count-weighted average of within-level r values. Primary inference:
    inverse-variance random-effects (forest_meta) combination of within-level Fisher-z values with
    standard error 1/sqrt(n - 3), back-transformed to r. Nothing ever pools raw trials across levels."""
    usable = [(s, w, lab) for s, w, lab in zip(level_stats, level_weights, labels)
              if s.get("status") == "computed"]
    if not usable:
        return {"status": "no_computable_level", "n_levels_attempted": len(level_stats)}
    combined: dict = {"n_levels_computed": len(usable), "n_levels_attempted": len(level_stats),
                       "levels_used": [lab for _, _, lab in usable]}
    for family, key in (("raw", "raw"), ("partialled", "partialled_after_recording_nuisances")):
        zs, ses, labs = [], [], []
        weighted_pairs: list[tuple[int, float]] = []
        for s, _w, lab in usable:
            entry = s[key]
            if entry.get("status") != "computed":
                continue
            n_lvl = s["n_clustering_units"]
            if n_lvl < 6:
                continue
            zs.append(_fisher_z(entry["r"]))
            ses.append(1.0 / np.sqrt(n_lvl - 3))
            labs.append(lab)
        for s, w, _lab in usable:
            entry = s[key]
            if entry.get("status") == "computed":
                weighted_pairs.append((w, entry["r"]))
        block: dict = {}
        if weighted_pairs:
            n_arr = np.array([n for n, _ in weighted_pairs], dtype=float)
            r_arr = np.array([r for _, r in weighted_pairs], dtype=float)
            block["trial_count_weighted_average_r"] = float(np.sum((n_arr / n_arr.sum()) * r_arr))
            block["trial_count_weighting_total"] = int(n_arr.sum())
        if len(zs) >= 2:
            meta = forest_meta(np.array(zs), np.array(ses), labs)
            block.update({
                "status": "computed",
                "fisher_z_combined_r": _fisher_back(meta["pooled"]),
                "fisher_z_combined_se": float(meta["se"]),
                "p_value": float(meta["p_value"]),
                "i_squared": meta.get("i_squared"),
                "n_levels_in_combination": len(zs)})
        elif len(zs) == 1:
            src = usable[0][0][key]
            block.update({"status": "computed", "fisher_z_combined_r": src["r"],
                          "p_value": src["p_value"], "n_levels_in_combination": 1})
        else:
            block.update({"status": "not_computable"})
        combined["raw" if key == "raw" else "partialled_after_recording_nuisances"] = block
    return combined


def classify_corpus_branch(cell: dict) -> dict:
    """Fixed-order corpus branch rule, declared before any result was read."""
    raw_q = cell.get("raw_benjamini_hochberg_q_value")
    part_q = cell.get("partialled_benjamini_hochberg_q_value")
    floor_block = cell.get("minimum_detectable_correlation_at_80pct_power", {}) or {}
    mdd = floor_block.get("minimum_detectable_correlation") if floor_block.get("status") == "computed" else None
    raw_sig = raw_q is not None and raw_q <= ALPHA
    part_sig = part_q is not None and part_q <= ALPHA
    effect = {
        "raw_effect_size_r": cell.get("raw_effect_size_r"),
        "partialled_effect_size_r": cell.get("partialled_effect_size_r"),
        "raw_q_value": raw_q, "partialled_q_value": part_q,
        "minimum_detectable_correlation_at_80pct_power": mdd,
        "standing_reference_r_units": REFERENCE_R_UNITS,
    }
    if part_sig:
        branch = "the_components_behavioural_signal_is_a_slow_between_session_state"
    elif raw_sig:
        branch = "the_between_session_association_is_explained_by_session_level_recording_nuisances"
    elif mdd is None or cell.get("status") != "computed":
        # A cell that could not run the declared partial, or whose unit count cannot support even
        # the power formula, is below its detection floor by construction -- never a powered null.
        branch = "inconclusive_below_detection_floor"
        effect["field_carrying_both_numbers"] = {
            "this_cell_minimum_detectable_correlation": mdd,
            "cell_status": cell.get("status"),
            "tested_against_reference_r": REFERENCE_R_UNITS,
            "reading": "the declared partial could not be computed at this clustering-unit count, so "
                       "the cell is below its own detection floor by construction; neither agreement "
                       "nor disagreement with any effect size is claimable"}
    elif (not raw_sig and not part_sig and mdd > REFERENCE_R_UNITS):
        branch = "inconclusive_below_detection_floor"
        effect["field_carrying_both_numbers"] = {
            "this_cell_minimum_detectable_correlation": mdd,
            "tested_against_reference_r": REFERENCE_R_UNITS,
            "reading": "a non-significant cell whose own minimum detectable correlation exceeds the "
                       "standing reference it is testing against; neither agreement nor disagreement "
                       "with any effect size is claimable at this resolution"}
    else:
        branch = "no_between_session_association_at_either_level"
    return {"branch": branch, **effect}


def classify_top_branch(cells: dict[str, dict], species_map: dict[str, str]) -> dict:
    """Fixed-order pooled verdict rule, declared before any result was read. ``cells`` maps corpus name
    (plus the optional '__pooled__' row) to its decision cell; ``species_map`` names each corpus's
    preparation so the non-human-only branch can attach every human floor alongside."""
    corpus_names = [k for k in cells if k != "__pooled__"]

    def q(cell: dict, family: str):
        v = cell.get(f"{family}_benjamini_hochberg_q_value")
        return v

    def sig(cell: dict, family: str) -> bool:
        v = q(cell, family)
        return v is not None and v <= ALPHA

    pooled = cells.get("__pooled__")
    pooled_part_sig = bool(pooled and sig(pooled, "partialled"))
    pooled_raw_sig = bool(pooled and sig(pooled, "raw"))
    pooled_sign = _sign(pooled.get("partialled_effect_size_r")) if pooled else None
    computable = [cells[name] for name in corpus_names
                  if cells[name].get("partialled_effect_size_r") is not None]
    signs_agree = pooled_sign is not None and bool(computable) and all(
        _sign(c["partialled_effect_size_r"]) == pooled_sign for c in computable)

    def attach(name: str) -> dict:
        cell = cells[name]
        out = {"branch": classify_corpus_branch(cell)["branch"]}
        for key in ("raw_effect_size_r", "partialled_effect_size_r",
                    "raw_benjamini_hochberg_q_value", "partialled_benjamini_hochberg_q_value"):
            out[key] = cell.get(key)
        floor_block = cell.get("minimum_detectable_correlation_at_80pct_power", {}) or {}
        out["minimum_detectable_correlation_at_80pct_power"] = floor_block.get("minimum_detectable_correlation") \
            if floor_block.get("status") == "computed" else None
        return out

    def per_corpus() -> dict:
        return {name: attach(name) for name in corpus_names}

    if pooled_part_sig and signs_agree:
        return {"branch": "the_components_behavioural_signal_is_a_slow_between_session_state",
                "per_corpus": per_corpus(), "pooled": attach("__pooled__")}
    if pooled_raw_sig and not pooled_part_sig:
        return {"branch": "the_between_session_association_is_explained_by_session_level_recording_nuisances",
                "per_corpus": per_corpus(), "pooled": attach("__pooled__")}
    any_non_human = any(sig(cells[n], fam) for n in corpus_names
                        if species_map.get(n) == "non_human" for fam in ("raw", "partialled"))
    any_human = any(sig(cells[n], fam) for n in corpus_names
                    if species_map.get(n) == "human" for fam in ("raw", "partialled"))
    if any_non_human and not any_human:
        human_floors = {n: {"minimum_detectable_correlation_at_80pct_power":
                            (cells[n].get("minimum_detectable_correlation_at_80pct_power", {}) or {})
                            .get("minimum_detectable_correlation"),
                            "standing_reference_r_units": REFERENCE_R_UNITS}
                        for n in corpus_names if species_map.get(n) == "human"}
        return {"branch": "the_between_session_association_is_present_in_non_human_corpora_only",
                "human_power_floors_stated_alongside": human_floors,
                "reading": "with the human floors alongside, a reader can tell a species difference "
                           "from a power difference",
                "per_corpus": per_corpus(), "pooled": attach("__pooled__")}
    nobody_sig = not any(sig(cells[n], fam) for n in corpus_names for fam in ("raw", "partialled"))

    def floor_of(name: str):
        block = cells[name].get("minimum_detectable_correlation_at_80pct_power", {}) or {}
        return block.get("minimum_detectable_correlation") if block.get("status") == "computed" else None

    floors = {n: floor_of(n) for n in corpus_names}
    all_below_floor = nobody_sig and floors and all(v is not None and v > REFERENCE_R_UNITS for v in floors.values())
    if all_below_floor:
        return {"branch": "inconclusive_below_detection_floor",
                "field_carrying_both_numbers_per_corpus": {
                    n: {"this_corpus_minimum_detectable_correlation": v,
                        "tested_against_reference_r": REFERENCE_R_UNITS} for n, v in floors.items()},
                "reading": "no corpus reached significance and every corpus's own minimum detectable "
                           "correlation exceeds the standing reference; neither agreement nor "
                           "disagreement is claimable at this resolution",
                "per_corpus": per_corpus(), "pooled": attach("__pooled__")}
    return {"branch": "no_between_session_association_at_either_level",
            "per_corpus": per_corpus(), "pooled": attach("__pooled__")}


# =======================================================================================================
# Corpus staging -- every corpus carrying both the component and a behavioural outcome
# =======================================================================================================

def _stop_staging(n_seen: int) -> bool:
    """True when a scope-limited run has seen its quota of clustering units; staging stops there so a
    smoke run never pays the full corpus's load time."""
    if SCOPE_LIMIT and n_seen >= int(SCOPE_LIMIT):
        _log(f"  scope limit active ({SCOPE_LIMIT}): staging stops after {n_seen} clustering units")
        return True
    return False


def load_macaque_single_item(root: Path) -> tuple[list[dict], list[dict], int]:
    """Sessions of the single-item macaque lateral prefrontal cortex corpus with the component, the
    binary outcome, total spike counts, unit count and series order (rank of the session within the
    corpus's chronologically ordered session list)."""
    rows, excluded = [], []
    paths = _macaque_session_paths(root)
    n_seen = 0
    for idx, path in enumerate(paths):
        if _stop_staging(n_seen):
            break
        n_seen += 1
        session = _macaque_load_session(path)
        counts = session["counts"]
        name = session["session"]
        if counts.shape[0] < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            excluded.append({"unit": name, "status": "excluded_too_few_trials",
                             "n_trials": int(counts.shape[0])})
            continue
        activity = counts.sum(axis=2)
        deviation = rate_free_state_deviation(activity)
        finite = np.isfinite(deviation)
        if int(finite.sum()) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            excluded.append({"unit": name, "status": "excluded_too_few_defined_direction_trials",
                             "n_defined": int(finite.sum())})
            continue
        covariates = trial_amplitude_covariates(counts)
        if covariates["status"] != "computed":
            excluded.append({"unit": name, "status": "excluded_amplitude_transform_not_computable"})
            continue
        rows.append({
            "unit": name, "animal": session["animal"],
            "deviation": np.asarray(deviation[finite], dtype=float),
            "worse_behaviour": (1.0 - session["is_corr"].astype(float))[finite],
            "spike_count": activity.sum(axis=1).astype(float)[finite],
            "unit_count": int(activity.shape[1]),
            "series_order": idx + 1,
            "n_error": int((~session["is_corr"].astype(bool)).sum()),
            "status": "ok",
        })
    return rows, excluded, n_seen


def load_watters_full_sessions(root: Path) -> tuple[list[dict], list[dict], int]:
    """Per-trial arrays plus unit count for every loadable behavioural session date of the multi-object
    macaque corpus. Completed per-session fits of the delivered multi-object analysis are served
    read-only when present, keeping the numbers bit-exact against that artifact; anything they lack is
    computed here under this module's own checkpoint directory."""
    loaded, refused = [], []
    n_seen = 0
    for session in iter_watters(root, bin_ms=BIN_MS):
        if _stop_staging(n_seen):
            break
        n_seen += 1
        if session["status"] != "loaded":
            refused.append({"unit": session["session"], "status": f"refused_{session['status']}"})
            continue
        # iter_watters has already materialised this session's counts, so the unit count comes from
        # the live session regardless of where the trial arrays come from.
        unit_count = int(session["counts"].shape[1])
        foreign = _read_foreign_fit(WATTERS_ARM_CHECKPOINT, f"watters_full_session|{session['session']}")
        if foreign is not None and "arrays" in foreign:
            arrays = {k: np.asarray(v) for k, v in foreign["arrays"].items()}
        else:
            def compute(s=session):
                arrays_new, _excluded, _usable = _observable_arrays(s["counts"], s)
                if arrays_new is None:
                    return None
                # Normalise to this module's worse-behaviour key; _observable_arrays names the graded
                # report column report_error.
                return {"deviation": arrays_new["deviation"],
                         "worse_behaviour": arrays_new["report_error"],
                         "spike_count": arrays_new["spike_count"],
                         "item_count": arrays_new["item_count"]}
            stored = _fit(f"watters_full|{session['session']}", compute)
            if stored is None:
                refused.append({"unit": session["session"],
                                "status": "refused_too_few_trials_with_defined_direction_and_report"})
                continue
            arrays = {k: np.asarray(v) for k, v in stored.items()}
        loaded.append({"session": session["session"], "animal": session.get("animal"),
                        "unit_count": unit_count, "arrays": arrays})
    return loaded, refused, n_seen


def _human_session_arrays(corpus: str, root: Path) -> tuple[list[dict], list[dict], int]:
    """Every admitted session of one human corpus with the component, outcome and nuisances. Trial
    correctness is carried as the outcome variable and is never an admission criterion; the only
    admission criteria are the corpus's own artifact flag where one exists, or a defined epoch onset
    where none does -- exactly what the shared admission iterators apply."""
    rows, excluded = [], []
    n_seen = 0
    for entry in ADMISSION_ITERATORS[corpus](root):
        if _stop_staging(n_seen):
            break
        n_seen += 1
        if not entry["usable_for_estimator"]:
            excluded.append({"unit": entry["session"], "patient": entry["patient"],
                             "status": f"excluded_{entry['exclusion_reason']}"})
            continue
        def compute(entry=entry):
            counts = delay_counts(entry["spike_lists"], entry["delay_onset"], entry["delay_window_s"],
                                  bin_ms=BIN_MS)
            activity_by_unit = counts.sum(axis=2)
            deviation = rate_free_state_deviation(activity_by_unit)
            finite = np.isfinite(deviation)
            return {
                "deviation": np.asarray(deviation[finite], dtype=float),
                "worse_behaviour": (1.0 - entry["is_correct"].astype(float))[finite],
                "spike_count": activity_by_unit.sum(axis=1).astype(float)[finite],
                "unit_count": int(activity_by_unit.shape[1]),
                "load_level": np.asarray(entry["load_level"])[finite].astype(int).tolist(),
            }
        arrays = _fit(f"human_session|{corpus}|{entry['session']}", compute)
        if len(arrays["deviation"]) < MIN_TRIALS_WITH_DEFINED_DIRECTION:
            excluded.append({"unit": entry["session"], "patient": entry["patient"],
                             "status": "excluded_too_few_defined_direction_trials"})
            continue
        rows.append({"unit": entry["session"], "patient": entry["patient"], **arrays})
    return rows, excluded, n_seen


def _participant_series_orders(rows: list[dict]) -> dict[str, int]:
    """Participant order within the corpus's recording series: the rank of the participant's earliest
    session among all of the corpus's admitted sessions ordered by identifier (identifiers embed the
    session number/date, so identifier order is chronological order)."""
    ids = sorted({r["unit"] for r in rows})
    rank_of_id = {sid: i + 1 for i, sid in enumerate(ids)}
    first: dict[str, int] = {}
    for r in rows:
        rk = rank_of_id[r["unit"]]
        cur = first.get(r["patient"])
        first[r["patient"]] = min(cur, rk) if cur is not None else rk
    return {p: i + 1 for i, p in enumerate(sorted(first, key=lambda p: first[p]))}


def human_patient_records_at_level(rows: list[dict], level: int | None = None) -> list[dict]:
    """One record per PARTICIPANT within one set-size level (or across all levels when ``level`` is
    None, used only for the within-participant statistic). Level-scoped covariates are computed on that
    level's own trials; unit count is the trial-weighted mean of contributing sessions' unit counts."""
    orders = _participant_series_orders(rows)
    records = []
    for patient in sorted({r["patient"] for r in rows}):
        parts = [r for r in rows if r["patient"] == patient]
        if level is not None:
            parts = [r for r in parts if level in set(int(v) for v in r["load_level"])]
        if not parts:
            continue
        masks = [np.array([int(v) for v in r["load_level"]]) == level for r in parts] \
            if level is not None else [np.ones(len(r["load_level"]), dtype=bool) for r in parts]
        # A fit served from an on-disk checkpoint has round-tripped through JSON, which turns array
        # fields into plain lists; coercion keeps both paths identical.
        deviation = np.concatenate([np.asarray(r["deviation"], dtype=float)[m] for r, m in zip(parts, masks)])
        worse = np.concatenate([np.asarray(r["worse_behaviour"], dtype=float)[m] for r, m in zip(parts, masks)])
        spikes = np.concatenate([np.asarray(r["spike_count"], dtype=float)[m] for r, m in zip(parts, masks)])
        n_per_part = [int(m.sum()) for m in masks]
        units = [r["unit_count"] for r in parts]
        unit_count = int(round(float(np.average(units, weights=n_per_part))))
        rec = build_unit_record(deviation, worse, unit_count, orders[patient], float(np.mean(spikes)),
                                item_count=level)
        rec["patient"] = patient
        rec["n_sessions"] = len(parts)
        records.append(rec)
    return records


def macaque_single_item_cells(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Records for the reachable sessions (the corpus's established error-trial floor, the same set the
    delivered between-session number was computed on) and the below-floor exclusions beside them."""
    records, below_floor = [], []
    for r in rows:
        if r["n_error"] < MIN_ERROR_TRIALS_FOR_REACHABILITY:
            below_floor.append({"unit": r["unit"], "status": "excluded_below_error_trial_reachability_floor",
                                "n_error": r["n_error"], "floor": MIN_ERROR_TRIALS_FOR_REACHABILITY})
            continue
        rec = build_unit_record(r["deviation"], r["worse_behaviour"], r["unit_count"], r["series_order"],
                                float(np.mean(r["spike_count"])))
        rec["unit"] = r["unit"]
        records.append(rec)
    return records, below_floor


def multi_object_level_cells(full_sessions: list[dict]) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Per item-count level: records for sessions reaching the corpus's per-level trial floor, and the
    per-session below-floor exclusions. Levels are never merged here."""
    levels = sorted({int(v) for f in full_sessions for v in f["arrays"]["item_count"].tolist()})
    cells: dict[str, list[dict]] = {str(lv): [] for lv in levels}
    excluded: dict[str, list[dict]] = {str(lv): [] for lv in levels}
    for animal in sorted({f["animal"] for f in full_sessions}):
        dates = sorted({f["session_date"] if "session_date" in f else f["session"].split("_", 1)[1]
                        for f in full_sessions if f["animal"] == animal})
        rank = {d: i + 1 for i, d in enumerate(dates)}
        for f in full_sessions:
            if f["animal"] != animal:
                continue
            date = f["session_date"] if "session_date" in f else f["session"].split("_", 1)[1]
            for lv in levels:
                mask = f["arrays"]["item_count"] == float(lv)
                n = int(mask.sum())
                if n < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
                    excluded[str(lv)].append({"unit": f["session"],
                                              "status": "excluded_below_level_trial_floor",
                                              "n_trials_at_level": n,
                                              "floor": MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION})
                    continue
                rec = build_unit_record(f["arrays"]["deviation"][mask], f["arrays"]["worse_behaviour"][mask],
                                        f["unit_count"], rank[date],
                                        float(np.mean(f["arrays"]["spike_count"][mask])), item_count=lv)
                rec["unit"] = f["session"]
                cells[str(lv)].append(rec)
    return cells, excluded


# =======================================================================================================
# Family-wide multiplicity, pooled row, verdicts
# =======================================================================================================

def _bh_family(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    correction = fdr_bh(np.array(p_values, dtype=float))
    return [float(qv) for qv in correction["q_values"]]


def finalize_families(per_corpus_blocks: list[dict]) -> tuple[dict, dict, dict]:
    """Benjamini-Hochberg within each family across computable corpora plus the pooled row; returns
    (pooled_block, decision_cells_by_name, q_assignment) with branches NOT yet attached."""
    computable = [b for b in per_corpus_blocks if b["cell"].get("status") == "computed"
                  and b["cell"].get("raw_p_value") is not None]

    def pooled_row(family: str) -> dict | None:
        zs, ses, labs = [], [], []
        for b in computable:
            r = b["cell"].get(f"{family}_fisher_z_combined_r", b["cell"].get(f"{family}_effect_size_r"))
            n = b["clustering_units_tested"]
            if r is None or n < 6:
                continue
            zs.append(_fisher_z(r))
            ses.append(1.0 / np.sqrt(n - 3))
            labs.append(b["corpus"])
        if len(zs) < 2:
            return None
        meta = forest_meta(np.array(zs), np.array(ses), labs)
        return {"status": "computed", "r": _fisher_back(meta["pooled"]), "se_fisher_z": float(meta["se"]),
                "p_value": float(meta["p_value"]), "i_squared": meta.get("i_squared"),
                "n_clustering_units_sum": int(sum(b["clustering_units_tested"] for b in computable))}

    raw_pool = pooled_row("raw")
    part_pool = pooled_row("partialled")

    raw_family = [b["cell"]["raw_p_value"] for b in computable] + ([raw_pool["p_value"]] if raw_pool else [])
    part_family = [b["cell"]["partialled_p_value"] for b in computable] + \
        ([part_pool["p_value"]] if part_pool else [])
    raw_qs = _bh_family(raw_family)
    part_qs = _bh_family(part_family)
    for b, qv in zip(computable, raw_qs[:len(computable)]):
        b["cell"]["raw_benjamini_hochberg_q_value"] = qv
    for b, qv in zip(computable, part_qs[:len(computable)]):
        b["cell"]["partialled_benjamini_hochberg_q_value"] = qv
    if raw_pool:
        raw_pool["benjamini_hochberg_q_value"] = raw_qs[len(computable)]
    if part_pool:
        part_pool["benjamini_hochberg_q_value"] = part_qs[len(computable)]

    cells_for_branch = {b["corpus"]: b["cell"] for b in per_corpus_blocks}
    pooled_cell = {
        "raw_effect_size_r": raw_pool["r"] if raw_pool else None,
        "partialled_effect_size_r": part_pool["r"] if part_pool else None,
        "raw_benjamini_hochberg_q_value": raw_pool.get("benjamini_hochberg_q_value") if raw_pool else None,
        "partialled_benjamini_hochberg_q_value": part_pool.get("benjamini_hochberg_q_value") if part_pool else None,
        "minimum_detectable_correlation_at_80pct_power": minimum_detectable_correlation(
            sum(b["clustering_units_tested"] for b in computable)),
    }
    cells_for_branch["__pooled__"] = pooled_cell
    pooled_block = {
        "raw": raw_pool, "partialled_after_recording_nuisances": part_pool,
        "minimum_detectable_correlation_at_80pct_power": pooled_cell[
            "minimum_detectable_correlation_at_80pct_power"],
        "combination_declaration": ("inverse-variance random-effects combination ON THE FISHER-z SCALE of "
                                     "each corpus's own primary within-corpus estimate; corpora are never "
                                     "merged at the trial level and level-structured corpora contribute "
                                     "only their within-level combined estimate"),
    }
    return pooled_block, cells_for_branch, {"computable": [b["corpus"] for b in computable]}


# =======================================================================================================
# Reproduction gate -- referenced artifacts read live; recomputed numbers compared bit-for-bit
# =======================================================================================================

def reproduction_gate(macaque_rows: list[dict], watters_loaded: list[dict]) -> dict:
    delivered = json.loads(REPRODUCTION_SOURCE_ARTIFACT.read_text())
    bias_only = json.loads(BIAS_ONLY_CONTROL_SOURCE_ARTIFACT.read_text())

    reachable = [r for r in macaque_rows if r["status"] == "ok"
                 and r["n_error"] >= MIN_ERROR_TRIALS_FOR_REACHABILITY]
    dev_means = np.array([float(np.mean(r["deviation"])) for r in reachable])
    worse_means = np.array([float(np.mean(r["worse_behaviour"])) for r in reachable])
    delivered_single = delivered["block_a"]["macaque_lPFC_single_item"]["between_session_association"]
    if len(reachable) >= 4:
        rng = np.random.default_rng(stable_seed("component_effect_size_and_anatomy|macaque|between_session"))
        got_single = pearson_permutation_test(dev_means, worse_means, n_perm=10000, rng=rng)
        single_checks = {
            "single_item_r_matches": bool(abs(got_single["r"] - delivered_single["r"]) <= REPRODUCTION_TOLERANCE),
            "single_item_p_matches": bool(abs(got_single["p_value"] - delivered_single["p_value"])
                                           <= REPRODUCTION_TOLERANCE),
        }
    else:
        # fewer reachable sessions than the delivered computation used -- only possible under an
        # active scope limit or a broken corpus; the gate reports rather than crashing
        got_single = None
        single_checks = {"single_item_r_matches": False, "single_item_p_matches": False,
                          "single_item_not_recomputable_too_few_reachable_sessions": True}

    watters_checks: dict[str, dict] = {}
    for row in delivered["block_a"]["macaque_multi_object"]["between_session_association"]:
        level = int(row["item_count"])
        dev_m, worse_m = [], []
        for f in watters_loaded:
            mask = f["arrays"]["item_count"] == float(level)
            if int(mask.sum()) < MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
                continue
            dev_m.append(float(np.mean(f["arrays"]["deviation"][mask])))
            worse_m.append(float(np.mean(f["arrays"]["worse_behaviour"][mask])))
        if len(dev_m) < 4:
            watters_checks[f"item_count_{level}"] = {"status": "not_recomputable_here",
                                                      "n_sessions_available": len(dev_m)}
            continue
        rng = np.random.default_rng(stable_seed(
            f"component_effect_size_and_anatomy|watters|between_session|{level}"))
        got = pearson_permutation_test(np.array(dev_m), np.array(worse_m), n_perm=10000, rng=rng)
        watters_checks[f"item_count_{level}"] = {
            "r_matches": bool(abs(got["r"] - row["between_session"]["r"]) <= REPRODUCTION_TOLERANCE),
            "p_matches": bool(abs(got["p_value"] - row["between_session"]["p_value"]) <= REPRODUCTION_TOLERANCE),
            "recomputed": {"r": got["r"], "p_value": got["p_value"]},
            "delivered": {"r": row["between_session"]["r"], "p_value": row["between_session"]["p_value"]}}

    swap_node = None
    node = bias_only.get("bias_only_control", {}).get("swap_primary|deviation|level2", {})
    if "bias_only_between_session" in node:
        swap_node = {"r": node["bias_only_between_session"]["r"],
                     "p_value": node["bias_only_between_session"]["p_value"]}

    checks = {
        "delivered_artifact_read_live": REPRODUCTION_SOURCE_ARTIFACT.exists(),
        "bias_only_control_artifact_read_live": BIAS_ONLY_CONTROL_SOURCE_ARTIFACT.exists(),
        **single_checks,
    }
    for key, val in watters_checks.items():
        checks[f"multi_object_{key}_matches"] = bool(val.get("r_matches") and val.get("p_matches")) \
            if "r_matches" in val else False
    bias_only_vs_recomputed = None
    if swap_node is not None and "item_count_2" in watters_checks \
            and "r_matches" in watters_checks["item_count_2"]:
        got2 = watters_checks["item_count_2"]["recomputed"]
        bias_only_vs_recomputed = {
            "bias_only_r": swap_node["r"], "recomputed_between_session_r": got2["r"],
            "absolute_difference": abs(swap_node["r"] - got2["r"]),
            "reading": ("the two statistics are computed on near-identical inputs; the swap control "
                        "restricts its trials to those where a swap is definable, so the two coincide "
                        "only up to that trial-subset difference and are not expected to match bit "
                        "for bit"),
        }
    return {
        "source_artifact": "results/component_effect_size_and_anatomy.json (read live, never modified)",
        "bias_only_control_source_artifact": ("results/swap_versus_imprecision_by_item_count.json "
                                               "(read live, never modified)"),
        "bias_only_control_item_count_2_delivered_live": swap_node,
        "bias_only_versus_recomputed_between_session_item_count_2": bias_only_vs_recomputed,
        "tolerance": REPRODUCTION_TOLERANCE,
        "checks": checks,
        "recomputed_single_item": ({"r": got_single["r"], "p_value": got_single["p_value"],
                                     "n_reachable_sessions": len(reachable)} if got_single is not None
                                    else {"n_reachable_sessions": len(reachable), "status": "not_recomputable"}),
        "recomputed_multi_object": watters_checks,
        "status": "reproduced_exactly" if all(checks.values()) else "not_reproduced",
    }


# =======================================================================================================
# Per-corpus assembly
# =======================================================================================================

SPECIES_BY_CORPUS = {
    "macaque_lPFC_single_item": "non_human",
    "macaque_multi_object": "non_human",
    "dandi_000469": "human",
    "dandi_001187": "human",
    "dandi_000574": "human",
}


def assemble_corpus(name: str, stats_or_combined: dict, level_labels: list[str] | None,
                    within_statistic: dict, exclusion_rows: list[dict], tested_units: int,
                    species: str, per_level_stats: dict | None = None) -> dict:
    """Attach the multiplicity-ready decision cell, the separated within/between statistics and the
    exclusion ledger to one corpus's computed statistics."""
    cell: dict = {"corpus": name, "species": species, "clustering_units_tested": tested_units}
    if level_labels is None:
        raw_entry = stats_or_combined.get("raw", {})
        part_entry = stats_or_combined.get("partialled_after_recording_nuisances", {})
        cell["status"] = stats_or_combined.get("status")
        if cell["status"] == "computed":
            cell["raw_effect_size_r"] = raw_entry.get("r")
            cell["partialled_effect_size_r"] = part_entry.get("r")
            cell["raw_p_value"] = raw_entry.get("p_value")
            cell["partialled_p_value"] = part_entry.get("p_value")
            cell["raw_fisher_z_combined_r"] = raw_entry.get("r")
            cell["partialled_fisher_z_combined_r"] = part_entry.get("r")
    else:
        raw_block = stats_or_combined.get("raw", {}) or {}
        part_block = stats_or_combined.get("partialled_after_recording_nuisances", {}) or {}
        cell["status"] = "computed" if raw_block.get("status") == "computed" else \
            stats_or_combined.get("status", "no_computable_level")
        if cell["status"] == "computed":
            cell["raw_effect_size_r"] = raw_block.get("trial_count_weighted_average_r")
            cell["partialled_effect_size_r"] = part_block.get("trial_count_weighted_average_r")
            cell["raw_p_value"] = raw_block.get("p_value")
            cell["partialled_p_value"] = part_block.get("p_value")
            cell["raw_fisher_z_combined_r"] = raw_block.get("fisher_z_combined_r")
            cell["partialled_fisher_z_combined_r"] = part_block.get("fisher_z_combined_r")
    cell["minimum_detectable_correlation_at_80pct_power"] = stats_or_combined.get(
        "minimum_detectable_correlation_at_80pct_power", minimum_detectable_correlation(tested_units))
    return {
        "corpus": name, "species": species, "clustering_units_tested": tested_units,
        "cell": cell,
        "within_session_statistic": {
            **within_statistic,
            "declaration": "reported separately; never pooled with the between-session statistics",
        },
        "between_session_statistics": stats_or_combined,
        "per_level_statistics": per_level_stats,
        "level_labels": level_labels,
        "exclusions": exclusion_rows,
        "estimator_declaration": MULTI_OBJECT_ESTIMATOR_DECLARATION if level_labels is not None and
        name == "macaque_multi_object" else
        ("associations computed within one set-size level and combined across levels; trials never "
         "pooled across levels" if level_labels is not None else
         "single clustering stratum; the generic between-unit estimator applies directly"),
    }


def finalize_branches(per_corpus_blocks: list[dict]) -> tuple[dict, dict]:
    pooled_block, cells_for_branch, meta = finalize_families(per_corpus_blocks)
    for b in per_corpus_blocks:
        b["verdict"] = classify_corpus_branch(b["cell"])
        del b["cell"]
    top = classify_top_branch(cells_for_branch, SPECIES_BY_CORPUS)
    return pooled_block, top


def within_vs_between_disagreement(within: dict, between_r: float | None, raw_between_r: float | None = None) -> str | None:
    """The between representative is the primary partialled estimate; where it is undefined the raw
    between estimate stands in. The two families are never merged into one number either way."""
    primary = between_r if between_r is not None else raw_between_r
    ws, bs = _sign(within.get("mean_value")), _sign(primary)
    if ws is None or bs is None or ws == 0 or bs == 0 or ws == bs:
        return None
    return ("the within-session and the between-session association disagree in sign; both are "
             "reported, the within-session association is the commensurable one, and the two are never "
             "pooled with each other")


# =======================================================================================================
# Driver
# =======================================================================================================

HUMAN_CORPORA = ("dandi_000469", "dandi_001187", "dandi_000574")


def main() -> None:
    t0 = time.time()
    root = data_root()
    output: dict = {
        "version": ANALYSIS_VERSION,
        "question": ("Is the accuracy-predicting component's behavioural signal a slow state that varies "
                      "between recording sessions (between participants, in humans), and does it survive "
                      "partialling session-level recording nuisances?"),
        "scope": {
            "corpora_staged": {
                "macaque_lPFC_single_item": {
                    "clustering_unit": "recording session",
                    "behavioural_outcome": "trial correctness, worse-behaviour convention",
                    "component": "rate-free direction deviation over the delay epoch"},
                "macaque_multi_object": {
                    "clustering_unit": "recording session, analysed within item-count level throughout",
                    "behavioural_outcome": "graded saccadic report deviation from the cued position",
                    "component": "rate-free direction deviation over the delay epoch"},
                "dandi_000469": {"clustering_unit": "participant",
                                  "behavioural_outcome": "trial correctness, image Sternberg",
                                  "component": "rate-free direction deviation over the maintenance epoch"},
                "dandi_001187": {"clustering_unit": "participant",
                                  "behavioural_outcome": "trial correctness, Sternberg",
                                  "component": "rate-free direction deviation over the maintenance epoch"},
                "dandi_000574": {"clustering_unit": "participant",
                                  "behavioural_outcome": "trial correctness, verbal Sternberg",
                                  "component": "rate-free direction deviation over the maintenance epoch"},
            },
            "corpora_excluded_by_name": {
                "inagaki_alm5": ("the shared loader carries instructed lick direction and condition labels "
                                  "but no per-trial correctness or error field, so this corpus does not "
                                  "carry the behavioural half"),
                "dandi_000574_eeg / dandi_000574_ieeg / dandi_000673": ("different recording grains; they do "
                                  "not carry the spike-grain direction-deviation component this analysis asks about"),
                "all other staged corpora": ("no pairing of this component with a working-memory accuracy "
                                              "outcome exists under any loader in this repository: stimulation "
                                              "outcomes, impulse protocols, pseudo-populations of neurons "
                                              "recorded separately, or no maintenance arm"),
            },
            "scope_limit_active": str(SCOPE_LIMIT) if SCOPE_LIMIT else None,
            "parameters": {
                "bin_ms": BIN_MS,
                "n_perm_primary": N_PERM_PRIMARY,
                "n_perm_patient_within_diagnostic": N_PERM_PATIENT_WITHIN,
                "min_trials_with_defined_direction": MIN_TRIALS_WITH_DEFINED_DIRECTION,
                "min_trials_for_behavioural_correlation": MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION,
                "min_error_trials_for_reachability": MIN_ERROR_TRIALS_FOR_REACHABILITY,
                "min_clustering_units_for_partial": MIN_CLUSTERING_UNITS_FOR_PARTIAL,
                "alpha": ALPHA, "power_target": POWER_TARGET,
                "standing_reference_r_units": REFERENCE_R_UNITS,
                "multiplicity": ("Benjamini-Hochberg within each family (raw, partialled) across the "
                                  "computable corpora plus the pooled row"),
                "primary_quality_tier_multi_object": PRIMARY_QUALITY_TIER,
            },
            "decision_rule_declared_before_reading_results": PRIMARY_DECISION_RULE_DECLARED_BEFORE_READING_RESULTS,
            "within_vs_between_declaration": WITHIN_VS_BETWEEN_DECLARATION,
            "seed_policy": ("every stochastic step draws from numpy default_rng seeded by zlib.crc32 of a "
                             "namespaced tag; the reproduction-gate tags reuse the delivered artifacts' own "
                             "tags verbatim so the recomputation is bit-exact"),
            "checkpoint_provenance": ("completed per-session multi-object fits written by the delivered "
                                       "multi-object analysis are served READ-ONLY for bit-exactness; this "
                                       "module writes only its own checkpoint directory"),
        },
        "status": "running",
    }
    _flush(output)

    # ---- staging ------------------------------------------------------------------------------------
    _log("staging macaque_lPFC_single_item")
    macaque_rows, macaque_excluded, macaque_seen = load_macaque_single_item(root)
    macaque_records, macaque_below_floor = macaque_single_item_cells(macaque_rows)
    _log(f"  -> {len(macaque_rows)} sessions with directions, {len(macaque_records)} reachable")

    _log("staging macaque_multi_object")
    watters_loaded, watters_refused, watters_seen = load_watters_full_sessions(root)
    _log(f"  -> {len(watters_loaded)} sessions usable, {len(watters_refused)} refused")

    human_rows: dict[str, list[dict]] = {}
    human_excluded: dict[str, list[dict]] = {}
    human_seen: dict[str, int] = {}
    for corpus in HUMAN_CORPORA:
        _log(f"staging {corpus}")
        rows, excl, seen = _human_session_arrays(corpus, root)
        human_rows[corpus], human_excluded[corpus], human_seen[corpus] = rows, excl, seen
        _log(f"  -> {len(rows)} admitted sessions, {len({r['patient'] for r in rows})} participants")

    output["_progress"] = {"staging_done": True,
                            "n_units": {"macaque_lPFC_single_item": len(macaque_records),
                                        "macaque_multi_object": len(watters_loaded),
                                        **{c: len(human_rows[c]) for c in HUMAN_CORPORA}}}
    _flush(output)

    # ---- reproduction gate ---------------------------------------------------------------------------
    _log("reproduction gate")
    gate = reproduction_gate(macaque_rows, watters_loaded)
    output["reproduction_gate"] = gate
    output.pop("_progress", None)
    _flush(output)
    _log(f"  -> {gate['status']}")
    if SCOPE_LIMIT and gate["status"] != "reproduced_exactly":
        gate["scope_limited_note"] = (
            "the run is scope-limited, so the recomputation cannot reproduce the delivered numbers "
            "computed on the full corpus; the gate is reported and does not stop a scope-limited run")
        output["reproduction_gate"] = gate
        _flush(output)
    elif gate["status"] != "reproduced_exactly":
        output["status"] = "stopped_reproduction_gate_failed"
        output["wall_clock_s"] = time.time() - t0
        _flush(output)
        _log("STOPPING: reproduction gate failed; no new number is reported")
        print(json.dumps({"status": output["status"], "checks": gate["checks"]}, indent=2))
        return

    # ---- per-corpus statistics -----------------------------------------------------------------------
    per_corpus_blocks: list[dict] = []

    _log("statistics: macaque_lPFC_single_item")
    single_stats = _fit(_k("stats|macaque_lPFC_single_item"),
                        lambda: between_session_stats(macaque_records, f"{MODULE_TAG}|macaque_lPFC_single_item"))
    reachable = [r for r in macaque_rows if r["n_error"] >= MIN_ERROR_TRIALS_FOR_REACHABILITY]
    within_vals = [float(np.corrcoef(r["deviation"], r["worse_behaviour"])[0, 1]) for r in reachable]
    within_single = _pool_values(within_vals)
    within_single["statistic"] = "per_session_point_r_between_deviation_and_worse_behaviour_pooled_by_sign_flip"
    within_single["minimum_detectable_paired_difference_at_80pct_power"] = \
        minimum_detectable_paired_difference(within_vals)
    blocks_single = assemble_corpus("macaque_lPFC_single_item", single_stats, None, within_single,
                                     macaque_excluded + macaque_below_floor, len(macaque_records), "non_human")
    per_corpus_blocks.append(blocks_single)

    _log("statistics: macaque_multi_object (within item-count level)")
    mo_cells, mo_excluded = multi_object_level_cells(watters_loaded)
    mo_level_stats, mo_labels, mo_weights = {}, [], []
    for lv in sorted(mo_cells, key=int):
        tag = f"{MODULE_TAG}|macaque_multi_object|item_count_{lv}"
        mo_level_stats[lv] = _fit(_k(f"stats|macaque_multi_object|{lv}"),
                                   lambda c=list(mo_cells[lv]), t=tag: between_session_stats(c, t))
        mo_labels.append(f"item_count_{lv}")
        mo_weights.append(int(sum(r["n_trials"] for r in mo_cells[lv])))
    mo_combined = _fit(_k("combined|macaque_multi_object"),
                       lambda: combine_levels(list(mo_level_stats.values()), mo_weights, mo_labels))
    arm_values = []
    for f in watters_loaded:
        foreign = _read_foreign_fit(WATTERS_ARM_CHECKPOINT, f"watters_deviation_arm|{f['session']}")
        value = None
        if foreign is not None:
            arm = foreign.get("arm", foreign)
            value = arm.get("within_load_trial_count_weighted", {}).get("raw_vs_report_error")
        if value is None:
            def compute(ff=f):
                tag = f"dissociation_replication_and_counting_noise|watters|{ff['session']}|{PRIMARY_QUALITY_TIER}"
                return _session_observable_arm(ff["arrays"], "deviation", tag)
            arm = _fit(f"watters_arm|{f['session']}", compute)
            value = arm.get("within_load_trial_count_weighted", {}).get("raw_vs_report_error")
        if value is not None:
            arm_values.append(value)
    within_mo = _pool_values(arm_values)
    blocks_mo = assemble_corpus(
        "macaque_multi_object", mo_combined, mo_labels, within_mo,
        [{"level": lv, **row} for lv, rows in mo_excluded.items() for row in rows] +
        [{"unit": r["unit"], "status": r["status"]} for r in watters_refused],
        len({r["unit"] for rows in mo_cells.values() for r in rows}), "non_human",
        per_level_stats={lv: mo_level_stats[lv] for lv in mo_level_stats})
    per_corpus_blocks.append(blocks_mo)

    for corpus in HUMAN_CORPORA:
        _log(f"statistics: {corpus} (participants, within set-size level)")
        rows = human_rows[corpus]
        load_levels = sorted({int(v) for r in rows for v in set(r["load_level"])})
        hu_level_stats, hu_labels, hu_weights = {}, [], []
        for lv in load_levels:
            lvl_records = _fit(_k(f"human_records|{corpus}|{lv}"),
                               lambda rr=rows, l=lv: human_patient_records_at_level(rr, l))
            tag = f"{MODULE_TAG}|{corpus}|set_size_{lv}"
            hu_level_stats[lv] = _fit(_k(f"stats|{corpus}|{lv}"),
                                       lambda lr=lvl_records, t=tag: between_session_stats(lr, t))
            hu_labels.append(f"set_size_{lv}")
            hu_weights.append(int(sum(r["n_trials"] for r in lvl_records)))
        combined = _fit(_k(f"combined|{corpus}"),
                        lambda ls=list(hu_level_stats.values()), ll=hu_labels, lw=hu_weights:
                        combine_levels(ls, lw, ll))
        patient_all = _fit(_k(f"human_records|{corpus}|all_levels"),
                           lambda rr=rows: human_patient_records_at_level(rr))
        within_hu = _fit(_k(f"within|{corpus}"), lambda rr=rows, c=corpus: _human_within_statistic(rr, c))
        blocks = assemble_corpus(corpus, combined, hu_labels, within_hu, human_excluded[corpus],
                                  len(patient_all), "human",
                                  per_level_stats={lv: hu_level_stats[lv] for lv in hu_level_stats})
        per_corpus_blocks.append(blocks)

    # ---- families, branches, accounting ---------------------------------------------------------------
    _log("multiplicity, branches, accounting")
    pooled_block, top_branch = finalize_branches(per_corpus_blocks)
    for b in per_corpus_blocks:
        b["within_versus_between_sign_disagreement"] = within_vs_between_disagreement(
            b["within_session_statistic"],
            b["verdict"].get("partialled_effect_size_r"),
            b["verdict"].get("raw_effect_size_r"))

    output["per_corpus"] = {b["corpus"]: b for b in per_corpus_blocks}
    output["pooled_across_corpora"] = pooled_block
    output["top_level_verdict"] = top_branch
    output["zero_drop_accounting"] = {
        "macaque_lPFC_single_item": {
            "n_sessions_on_disk": macaque_seen,
            "n_sessions_with_directions": len(macaque_rows),
            "n_sessions_reaching_the_reachability_floor": len(macaque_records),
            "n_excluded_too_few_trials": sum(1 for e in macaque_excluded),
            "n_excluded_below_reachability_floor": len(macaque_below_floor),
            "reconciles": bool(macaque_seen ==
                               len(macaque_rows) + len(macaque_excluded) and
                               len(macaque_rows) == len(macaque_records) + len(macaque_below_floor)),
        },
        "macaque_multi_object": {
            "n_behavioural_session_dates_seen": watters_seen,
            "n_sessions_usable": len(watters_loaded),
            "n_sessions_refused": len(watters_refused),
            "refusals_by_reason": {reason: sum(1 for r in watters_refused if r["status"] == reason)
                                    for reason in sorted({r["status"] for r in watters_refused})},
            "n_sessions_contributing_any_level": len({r["unit"] for rows in mo_cells.values() for r in rows}),
            "reconciles": bool(watters_seen == len(watters_loaded) + len(watters_refused)),
        },
        **{corpus: {
            "n_sessions_seen": human_seen[corpus],
            "n_sessions_admitted": len(human_rows[corpus]),
            "n_sessions_excluded": len(human_excluded[corpus]),
            "exclusion_reason_tally": {
                reason: sum(1 for e in human_excluded[corpus] if e["status"] == reason)
                for reason in sorted({e["status"] for e in human_excluded[corpus]})},
            "n_participants": len(patient_all),
            "reconciles": bool(human_seen[corpus] == len(human_rows[corpus]) + len(human_excluded[corpus])),
        } for corpus, patient_all in ((c, _fit(_k(f"human_records|{c}|all_levels"),
                                               lambda rr=human_rows[c]: human_patient_records_at_level(rr)))
                                      for c in HUMAN_CORPORA)},
    }
    output["accounting_reconciles_everywhere"] = all(
        v["reconciles"] for v in output["zero_drop_accounting"].values())

    # Standing artifact discipline: the scope block carries the session counts, the exclusion ledger,
    # and the wall clock alongside the parameters declared above.
    output["scope"]["session_accounting_summary"] = {
        name: {k: v for k, v in block.items() if k != "exclusion_reason_tally"}
        for name, block in output["zero_drop_accounting"].items()}
    output["scope"]["exclusion_ledger_by_corpus"] = {
        "macaque_lPFC_single_item": macaque_excluded + macaque_below_floor,
        "macaque_multi_object": [{"level": lv, **row} for lv, rows in mo_excluded.items() for row in rows]
                                 + [{"unit": r["unit"], "status": r["status"]} for r in watters_refused],
        **{c: human_excluded[c] for c in HUMAN_CORPORA}}
    output["scope"]["wall_clock_s"] = time.time() - t0
    output["scope"]["deterministic_seed_example"] = {
        "tag": f"{MODULE_TAG}|seed_example", "seed": int(stable_seed(f"{MODULE_TAG}|seed_example"))}

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)
    print(json.dumps({
        "reproduction_gate": gate["status"],
        "top_level_branch": top_branch["branch"],
        "per_corpus_branches": {b["corpus"]: b["verdict"]["branch"] for b in per_corpus_blocks},
        "accounting_reconciles_everywhere": output["accounting_reconciles_everywhere"],
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2))


def _human_within_statistic(rows: list[dict], corpus: str) -> dict:
    """Per-participant within-participant association: one point correlation over ALL of that
    participant's admitted trials (every set-size level together -- this is the within-participant
    counterpart of the participant-mean statistic and is never pooled with any between-unit number),
    pooled across participants by the paired sign-flip test."""
    values = []
    for patient in sorted({r["patient"] for r in rows}):
        parts = [r for r in rows if r["patient"] == patient]
        x = np.concatenate([np.asarray(r["deviation"], dtype=float) for r in parts])
        y = np.concatenate([np.asarray(r["worse_behaviour"], dtype=float) for r in parts])
        if len(x) < MIN_TRIALS_PER_PARTICIPANT_WITHIN or y.std() == 0.0 or x.std() == 0.0:
            continue
        rng = np.random.default_rng(stable_seed(f"{MODULE_TAG}|{corpus}|within|{patient}"))
        entry = partial_correlation_permutation_test(y, x, [], N_PERM_PATIENT_WITHIN, rng)
        if entry.get("status") == "computed":
            values.append(entry["r"])
    pooled = slope_across_sessions_test(values, alternative="two-sided") if values else {
        "status": "not_computable", "n_participants": 0}
    pooled["statistic"] = "per_participant_point_r_over_all_admitted_trials_pooled_by_sign_flip"
    pooled["n_participants"] = len(values)
    pooled["minimum_detectable_paired_difference_at_80pct_power"] = \
        minimum_detectable_paired_difference(values) if len(values) >= 2 else {"status": "not_computable"}
    return pooled


if __name__ == "__main__":
    main()
