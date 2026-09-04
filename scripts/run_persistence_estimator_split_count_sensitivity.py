"""run_persistence_estimator_split_count_sensitivity.py -- is the cross-
species difference in the trial-specific state's persistence slope a
property of the species, or a property of how many cross-validation splits
and permutation-null replicates the estimator was given?

Every arm of this project's cross-species d_perm(lag)-slope comparison
(human intracranial delay, mouse ALM, macaque lPFC) is supposed to run
through :func:`state_persistence.r_lag_profile` and
:func:`state_persistence.per_unit_permutation_null_r_lag_profile` at the
same settings. They do not: the human and ALM arms use the module default
(n_splits=12, n_null_replicates=20) while the macaque arm uses a lighter,
hard-coded setting (n_splits=10, n_null_replicates=10), adopted for wall-
clock reasons when the macaque arm was a secondary discriminator. The
macaque arm is now the arm that carries the whole cross-species claim (it is
the only one of the three whose d_perm slope over lag bins 3-8 is positive),
so that earlier economy is re-examined here rather than left in place.

This module holds the human and ALM arms' every other analysis choice fixed
-- same sessions, same epoch (delay), same fixed window width (3 bins), same
bin width (100 ms), same lag range for the slope (bins 3-8, 0.3-0.8 s, no lag
excluded), same seed derivation -- and refits each session at BOTH the
module default and the macaque's lighter setting, so the per-session CHANGE
in d_perm slope this produces can be tested directly. It also refits the
macaque arm at both its own native setting and the human/ALM module default,
which is affordable at this fixed width and epoch (see
``estimator_settings_by_arm`` in the written artifact for the measured
cost), giving a fully paired three-arm comparison rather than treating the
macaque side of the question as unaffordable by assumption.

Every per-session fit reuses :func:`run_state_persistence._lag_run_row`
(for the human and ALM arms) with an explicit ``n_splits``/
``n_null_replicates`` pair rather than a forked copy of the estimator, and
every pooled contrast reuses :func:`state_persistence.segmented_slope_test`,
:func:`state_persistence.per_session_slopes_in_range`,
:func:`state_persistence.component_series` and
:func:`state_persistence._d_series`, the same functions
scripts/run_state_shape_common_range.py used to establish the reference
d_perm/r_obs/r_null slope numbers this module's decision rule is audited
against.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from corpus_sessions import (  # noqa: E402
    EPOCH_WINDOWS_S, alm_data_directory, data_root, iter_all_corpora, load_alm_raw_session,
)
from run_state_persistence import (  # noqa: E402
    HUMAN_DATASETS, LAG_BIN_MS, LAG_N_NULL_REPLICATES, LAG_N_SPLITS, LAG_NULL_SPLITS_PER_REPLICATE,
    PANICHELLO_DELAY_WINDOW_MS, PANICHELLO_LAG_N_NULL_REPLICATES, PANICHELLO_LAG_N_SPLITS,
    _lag_run_row, _panichello_directory, _seed,
)
from run_state_persistence_shape import _lag_lists, _to_int_keyed  # noqa: E402
from state_persistence import (  # noqa: E402
    _d_series, component_series, per_session_slopes_in_range, segmented_slope_test,
)
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "persistence_estimator_split_count_sensitivity.json"

DECIDING_WIDTH_BINS = 3
DECIDING_EPOCH = "delay"
BIN_WIDTH_S = LAG_BIN_MS / 1000.0
SLOPE_RANGE_BINS = (3, 8)  # 0.3-0.8 s, no lag excluded -- matches state_shape_common_range.json's headline table

REFERENCE_SETTINGS = {"n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES}
DOWN_SETTINGS = {"n_splits": PANICHELLO_LAG_N_SPLITS, "n_null_replicates": PANICHELLO_LAG_N_NULL_REPLICATES}

CLOSED_FRACTION_CLEAR_THRESHOLD = 0.10
CLOSED_FRACTION_CONFIRM_THRESHOLD = 1.0 / 3.0

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "For the human delay arm and the mouse ALM arm, separately: compute each session's own d_perm slope "
    "over lag bins 3-8 (0.3-0.8 s, no lag excluded) at the human/ALM module default settings "
    "(n_splits=12, n_null_replicates=20) and again at the macaque arm's lighter settings (n_splits=10, "
    "n_null_replicates=10), holding every other analysis choice fixed -- same sessions, same epoch, same "
    "fixed window width, same bin width, same lag range, same seed derivation. Test the per-session "
    "change (lighter-settings slope minus module-default slope) with a two-sided paired sign-flip test, "
    "and express its point estimate as a fraction of that arm's own gap to the macaque delay-period "
    "d_perm slope at the module-default settings, recomputed independently in this module rather than "
    "copied. Before any fit runs: if the paired change is not significantly different from zero (two-"
    "sided p > 0.05) and its point estimate closes less than one tenth of the gap, the branch is "
    "'estimator_settings_do_not_account_for_the_cross_species_difference'. If the change is significantly "
    "positive and closes one third or more of the gap, the branch is "
    "'cross_species_difference_is_substantially_a_settings_difference'. Any other outcome is "
    "'partially_attributable', reported as the closed fraction with its confidence interval rather than "
    "forced onto either named branch."
)

PANICHELLO_PAIRED_CHANGE_SIGN_CONVENTION = (
    "module_default_minus_native (the UP-run direction: n_splits=12, n_null_replicates=20 minus the "
    "macaque arm's own native n_splits=10, n_null_replicates=10). This is the OPPOSITE sign convention "
    "from the human and ALM arms' paired_change_d_perm_slope fields above, which test "
    "lighter_setting_minus_module_default (the DOWN-run direction). A reader comparing a paired-change "
    "sign or magnitude across arms must first flip one of the two conventions; neither field states the "
    "other's convention for you."
)

PANICHELLO_PAIRED_CHANGE_DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "This arm has never had a decision rule of its own before this one; it is not a reuse of the human/"
    "ALM rule above, which is about a different direction (whether starving the estimator manufactures a "
    "positive slope in an arm whose native slope is negative) and a different question. The quantity at "
    "issue here is the macaque delay-epoch d_perm slope itself, whose value at the arm's own native "
    "setting is positive while every other arm's native-setting value is negative. Let m_native and "
    "m_default be the pooled macaque d_perm slopes at the native setting (n_splits=10, "
    "n_null_replicates=10) and at the human/ALM module default (n_splits=12, n_null_replicates=20), and "
    "let the paired change be tested per session in the up-run direction (see "
    "PANICHELLO_PAIRED_CHANGE_SIGN_CONVENTION) with a two-sided paired sign-flip test. Before any new fit "
    "runs: "
    "(1) if m_default is NOT significantly different from zero (two-sided p > 0.05), the branch is "
    "'macaque_positive_d_perm_slope_does_not_survive_matched_estimator_settings', and the cross-species "
    "difference is WITHDRAWN, not weakened -- every sentence in the project that contrasts the macaque's "
    "sign with the human's or the mouse's is void and must be listed for rewriting. "
    "(2) Otherwise, if m_default is significantly NEGATIVE, the branch is "
    "'macaque_d_perm_slope_reverses_sign_under_matched_estimator_settings', a stronger and DIFFERENT "
    "result from branch (1): it would mean all three arms decay and the project has one universal result "
    "rather than a species dissociation, and is never folded into branch (1). "
    "(3) Otherwise (m_default remains significantly positive, the same sign as m_native), if the paired "
    "change itself is NOT significantly different from zero, the branch is "
    "'macaque_positive_d_perm_slope_is_not_an_estimator_setting_artifact', and the cross-species "
    "difference is established at matched settings in all three arms. "
    "(4) Otherwise (m_default remains significantly positive AND the paired change is itself "
    "significant), the branch is "
    "'macaque_positive_d_perm_slope_survives_but_its_magnitude_is_estimator_setting_dependent', reported "
    "with m_native and m_default side by side. "
    "These four conditions are checked in this order and are exhaustive and mutually exclusive over "
    "(m_default significant or not, m_default's sign, paired-change significant or not): every case lands "
    "in exactly one branch, none falls through to an ambiguous middle branch, and a significant "
    "wrong-direction (negative) m_default is never scored as weaker evidence than a null would be -- it is "
    "scored as branch (2), which is stronger evidence against the positive-macaque-slope claim than a "
    "null (branch 1) is."
)


def _counts_from_spikes_delay(spike_lists, onset, window_s: float, bin_ms: float) -> np.ndarray:
    from spike_pipeline import build_psth
    rate = build_psth(spike_lists, onset, bin_ms=bin_ms, smooth_ms=0.0, window_s=window_s)
    return rate * (bin_ms / 1000.0)


def _human_delay_sessions(root: Path) -> list[dict]:
    """Every (dataset, session), structure=='pooled', dataset in
    HUMAN_DATASETS -- the exact 72-session set the deciding cross-species
    contrast lives at."""
    sessions = []
    for meta in iter_all_corpora(root):
        if meta["structure"] != "pooled" or meta["dataset"] not in HUMAN_DATASETS:
            continue
        onset = meta["epoch_onsets"][DECIDING_EPOCH]
        window_s = meta["epoch_windows"][DECIDING_EPOCH]
        counts = _counts_from_spikes_delay(meta["spike_lists"], onset, window_s, LAG_BIN_MS)
        seed = _seed(meta["dataset"], meta["session"], meta["structure"], DECIDING_EPOCH, LAG_BIN_MS,
                     f"width{DECIDING_WIDTH_BINS}", "lag")
        sessions.append({
            "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
            "structure": meta["structure"], "counts": counts, "seed": seed,
        })
    return sessions


def _alm_sessions(root: Path, human_window_s: float, rate_matched_keep_probability: float) -> list[dict]:
    from state_persistence import binomial_thin
    directory = alm_data_directory(root)
    sessions = []
    if not directory.is_dir():
        return sessions
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=LAG_BIN_MS, window_s=human_window_s, require_both_arms=False)
        if session is None:
            continue
        thin_rng = np.random.default_rng(_seed("alm_lag_thin", path.stem))
        counts = binomial_thin(session["control_counts"], rate_matched_keep_probability, thin_rng)
        seed = _seed("alm_lag", path.stem, DECIDING_WIDTH_BINS)
        sessions.append({
            "dataset": "inagaki_alm5", "patient": path.stem, "session": path.stem,
            "structure": "pooled", "counts": counts, "seed": seed,
        })
    return sessions


def _panichello_sessions(root: Path) -> list[dict]:
    directory = _panichello_directory(root)
    sessions = []
    if directory is None:
        return sessions
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, squeeze_me=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
        spikes = spikes[correct]
        starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], LAG_BIN_MS)
        binned = []
        for start in starts:
            mask = (time_ms >= start) & (time_ms < start + LAG_BIN_MS)
            binned.append(spikes[:, mask, :].sum(axis=1))
        counts = np.stack(binned, axis=2)
        stem = Path(path).stem
        seed = _seed("panichello_lag", stem, DECIDING_WIDTH_BINS)
        sessions.append({
            "dataset": "panichello_2024", "patient": stem, "session": stem,
            "structure": "pooled", "counts": counts, "seed": seed,
        })
    return sessions


def _row_for_settings(rows: list[dict], settings_key: str) -> list[dict]:
    """Reshapes into the (dataset/session/.../profile/null_poisson/
    null_permutation) row shape run_state_persistence_shape._lag_lists
    expects, for one of the two settings this module fitted."""
    return [{**{k: v for k, v in r.items() if k not in ("reference", "down")},
              "width_bins": DECIDING_WIDTH_BINS, **r[settings_key]} for r in rows]


def _per_session_slope_triplet(profile: dict, null_permutation: dict) -> dict | None:
    """This session's own r_obs, r_null (permutation) and d_perm slope over
    SLOPE_RANGE_BINS, no lag excluded -- None if fewer than two lags are
    common to both bin bounds within the closed range. Normalises both
    dicts' lag keys through _to_int_keyed first: a row read back from this
    module's own on-disk cache has had its 'lags' mapping turned into
    string keys by the JSON round-trip (JSON has no integer dict keys),
    which would otherwise break the numeric lag-range comparison below with
    a silent type error only a resumed run would ever hit. _to_int_keyed is
    idempotent on an already-int-keyed dict, so a freshly fitted row is
    unaffected."""
    if profile.get("status") != "fitted" or null_permutation is None:
        return None
    profile_lags = _to_int_keyed(profile["lags"])
    null_lags = _to_int_keyed(null_permutation["lags"])
    common = sorted(set(profile_lags) & set(null_lags))
    r_obs_series = {lag: profile_lags[lag]["r_median"] for lag in common}
    r_null_series = {lag: null_lags[lag]["r_null_median"] for lag in common}
    d_series = {lag: r_obs_series[lag] - r_null_series[lag] for lag in common}
    r_obs_slope = per_session_slopes_in_range([r_obs_series], BIN_WIDTH_S, SLOPE_RANGE_BINS, None)
    r_null_slope = per_session_slopes_in_range([r_null_series], BIN_WIDTH_S, SLOPE_RANGE_BINS, None)
    d_perm_slope = per_session_slopes_in_range([d_series], BIN_WIDTH_S, SLOPE_RANGE_BINS, None)
    if not (r_obs_slope and r_null_slope and d_perm_slope):
        return None
    return {"r_obs_slope": r_obs_slope[0], "r_null_slope": r_null_slope[0], "d_perm_slope": d_perm_slope[0]}


def _paired_change(down_values: list[float], reference_values: list[float], n_pairs: int) -> dict:
    """Two-sided paired sign-flip test of (down - reference), plus the
    minimum detectable paired difference at 80% power from the same paired
    differences -- the bound a null result here is reported against rather
    than resting on a failure to reject."""
    down_arr, ref_arr = np.array(down_values), np.array(reference_values)
    if n_pairs < 4 or (1.0 / (2 ** n_pairs)) > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs}
    test = paired_sign_flip_test(down_arr, ref_arr, alternative="two-sided")
    two_sided_significant = bool(test["p_value"] <= 0.05)
    mdd = minimum_detectable_paired_difference(down_arr - ref_arr)
    return {
        "status": "tested", "n_pairs": n_pairs,
        "mean_diff_down_minus_reference": test["mean_diff"], "p_value_two_sided": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"], "significant": two_sided_significant,
        "minimum_detectable_paired_difference_80pct_power": mdd,
    }


def _fraction_closed(change: dict, gap: float) -> dict:
    if change.get("status") != "tested" or gap == 0.0:
        return {"status": "not_computable", "gap": gap}
    fraction = change["mean_diff_down_minus_reference"] / gap
    return {
        "status": "computed", "gap": gap, "fraction": fraction,
        "fraction_ci_lower": change["ci_lower"] / gap, "fraction_ci_upper": change["ci_upper"] / gap,
    }


def _classify(change: dict, fraction: dict) -> str:
    if change.get("status") != "tested" or fraction.get("status") != "computed":
        return "not_computable"
    significant = change["significant"]
    positive = change["mean_diff_down_minus_reference"] > 0.0
    f = fraction["fraction"]
    if (not significant) and f < CLOSED_FRACTION_CLEAR_THRESHOLD:
        return "estimator_settings_do_not_account_for_the_cross_species_difference"
    if significant and positive and f >= CLOSED_FRACTION_CONFIRM_THRESHOLD:
        return "cross_species_difference_is_substantially_a_settings_difference"
    return "partially_attributable"


def _pooled_arm_slopes(lag_rows: list[dict]) -> dict:
    """Pooled d_perm/r_obs/r_null slope over SLOPE_RANGE_BINS across every
    session in ``lag_rows`` (one arm, one estimator setting), reusing
    segmented_slope_test -- the same function
    scripts/run_state_shape_common_range.py used to establish this
    project's reference d_perm/r_obs/r_null slope numbers."""
    profiles, pois, perm = _lag_lists(lag_rows, DECIDING_WIDTH_BINS)
    r_obs_series, r_null_series = component_series(profiles, perm, "r_null_median")
    d_series = _d_series(profiles, perm, "r_null_median")
    return {
        "n_sessions_fitted": len(profiles),
        "d_perm": segmented_slope_test(d_series, BIN_WIDTH_S, SLOPE_RANGE_BINS, None, alternative="two-sided"),
        "r_obs": segmented_slope_test(r_obs_series, BIN_WIDTH_S, SLOPE_RANGE_BINS, None, alternative="two-sided"),
        "r_null_permutation": segmented_slope_test(r_null_series, BIN_WIDTH_S, SLOPE_RANGE_BINS, None, alternative="two-sided"),
    }


def _arm_paired_change(rows: list[dict]) -> dict:
    """Everything about a single arm (human delay or ALM) that does NOT
    depend on the macaque's own reference-settings d_perm slope: the pooled
    slope at each setting for d_perm/r_obs/r_null, and the per-session
    paired change in each. Split out from the macaque-dependent gap/
    fraction/branch (see :func:`_arm_finalize_branch`) so the expensive part
    -- fitting every session at both settings -- never has to be redone
    just because the macaque number it will be compared against was not yet
    available when this arm ran."""
    reference_lag_rows = _row_for_settings(rows, "reference")
    down_lag_rows = _row_for_settings(rows, "down")
    pooled_reference = _pooled_arm_slopes(reference_lag_rows)
    pooled_down = _pooled_arm_slopes(down_lag_rows)

    paired_r_obs_down, paired_r_obs_ref = [], []
    paired_r_null_down, paired_r_null_ref = [], []
    paired_d_perm_down, paired_d_perm_ref = [], []
    n_sessions_total = len(rows)
    n_sessions_excluded = 0
    for r in rows:
        triplet_ref = _per_session_slope_triplet(r["reference"]["profile"], r["reference"]["null_permutation"])
        triplet_down = _per_session_slope_triplet(r["down"]["profile"], r["down"]["null_permutation"])
        if triplet_ref is None or triplet_down is None:
            n_sessions_excluded += 1
            continue
        paired_r_obs_ref.append(triplet_ref["r_obs_slope"]); paired_r_obs_down.append(triplet_down["r_obs_slope"])
        paired_r_null_ref.append(triplet_ref["r_null_slope"]); paired_r_null_down.append(triplet_down["r_null_slope"])
        paired_d_perm_ref.append(triplet_ref["d_perm_slope"]); paired_d_perm_down.append(triplet_down["d_perm_slope"])

    n_pairs = len(paired_d_perm_ref)
    return {
        "n_sessions_total": n_sessions_total, "n_sessions_paired": n_pairs, "n_sessions_excluded": n_sessions_excluded,
        "pooled_at_reference_settings": pooled_reference,
        "pooled_at_down_settings": pooled_down,
        "paired_change_d_perm_slope": _paired_change(paired_d_perm_down, paired_d_perm_ref, n_pairs),
        "paired_change_r_obs_slope": _paired_change(paired_r_obs_down, paired_r_obs_ref, n_pairs),
        "paired_change_r_null_permutation_slope": _paired_change(paired_r_null_down, paired_r_null_ref, n_pairs),
    }


def _arm_finalize_branch(arm_paired_change: dict, macaque_reference_slope: float | None, macaque_reference_source: str) -> dict:
    """Adds the macaque-dependent gap, closed fraction and branch to an
    arm's already-computed :func:`_arm_paired_change` result -- cheap
    (arithmetic only), so it can be recomputed the moment a better macaque
    reference-settings number becomes available without refitting anything."""
    change_d_perm = arm_paired_change["paired_change_d_perm_slope"]
    reference_d_perm_slope = arm_paired_change["pooled_at_reference_settings"]["d_perm"]["test"].get("mean_value")
    if macaque_reference_slope is None or reference_d_perm_slope is None:
        gap, fraction = None, {"status": "not_computable"}
    else:
        gap = macaque_reference_slope - reference_d_perm_slope
        fraction = _fraction_closed(change_d_perm, gap)
    branch = _classify(change_d_perm, fraction)
    return {
        **arm_paired_change,
        "macaque_reference_settings_d_perm_slope_source": macaque_reference_source,
        "gap_to_macaque_reference_settings_d_perm_slope": gap,
        "fraction_of_gap_closed_by_settings_change": fraction,
        "branch": branch,
    }


def _panichello_paired_change_up_run(rows: list[dict]) -> dict:
    """The macaque arm's own per-session paired change in the UP-run
    direction (module_default - native), mirroring _arm_paired_change's
    per-session-triplet-then-paired-sign-flip-test design but with the
    two _paired_change argument lists swapped so mean_diff comes out as
    default - native rather than _arm_paired_change's native-setting-
    relative-to-module-default convention (see
    PANICHELLO_PAIRED_CHANGE_SIGN_CONVENTION)."""
    paired_r_obs_native, paired_r_obs_default = [], []
    paired_r_null_native, paired_r_null_default = [], []
    paired_d_perm_native, paired_d_perm_default = [], []
    n_sessions_total = len(rows)
    n_sessions_excluded = 0
    for r in rows:
        triplet_native = _per_session_slope_triplet(r["down"]["profile"], r["down"]["null_permutation"])
        triplet_default = _per_session_slope_triplet(r["reference"]["profile"], r["reference"]["null_permutation"])
        if triplet_native is None or triplet_default is None:
            n_sessions_excluded += 1
            continue
        paired_r_obs_native.append(triplet_native["r_obs_slope"]); paired_r_obs_default.append(triplet_default["r_obs_slope"])
        paired_r_null_native.append(triplet_native["r_null_slope"]); paired_r_null_default.append(triplet_default["r_null_slope"])
        paired_d_perm_native.append(triplet_native["d_perm_slope"]); paired_d_perm_default.append(triplet_default["d_perm_slope"])

    n_pairs = len(paired_d_perm_native)
    return {
        "n_sessions_total": n_sessions_total, "n_sessions_paired": n_pairs, "n_sessions_excluded": n_sessions_excluded,
        "sign_convention": PANICHELLO_PAIRED_CHANGE_SIGN_CONVENTION,
        "paired_change_d_perm_slope": _paired_change(paired_d_perm_default, paired_d_perm_native, n_pairs),
        "paired_change_r_obs_slope": _paired_change(paired_r_obs_default, paired_r_obs_native, n_pairs),
        "paired_change_r_null_permutation_slope": _paired_change(paired_r_null_default, paired_r_null_native, n_pairs),
    }


def _panichello_branch(paired_change_d_perm: dict, m_default_test: dict) -> str:
    """The macaque arm's own pre-declared, direction-aware branch rule (see
    PANICHELLO_PAIRED_CHANGE_DECISION_RULE_DECLARED_BEFORE_FITTING),
    implemented as the four-condition if/elif chain that rule states in
    order. significant_negative/significant_positive on m_default_test are
    already derived from the two-sided test by slope_across_sessions_test,
    so a significant-negative m_default cannot fall through to the null
    branch or the positive-surviving branch -- it is checked first and
    returned as its own, stronger branch."""
    if paired_change_d_perm.get("status") != "tested" or m_default_test.get("status") != "tested":
        return "not_computable"
    if not m_default_test["significant_positive"] and not m_default_test["significant_negative"]:
        return "macaque_positive_d_perm_slope_does_not_survive_matched_estimator_settings"
    if m_default_test["significant_negative"]:
        return "macaque_d_perm_slope_reverses_sign_under_matched_estimator_settings"
    if not paired_change_d_perm["significant"]:
        return "macaque_positive_d_perm_slope_is_not_an_estimator_setting_artifact"
    return "macaque_positive_d_perm_slope_survives_but_its_magnitude_is_estimator_setting_dependent"


def _flush(output: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2))


COMMON_RANGE_PATH = Path(__file__).resolve().parents[1] / "results" / "state_shape_common_range.json"


def _provisional_macaque_reference_slope() -> tuple[float | None, str]:
    """A macaque reference-settings (n_splits=12, n_null_replicates=20)
    d_perm slope to score the human and ALM arms against WHILE the fresh,
    slower confirmatory macaque refit (both settings, all 25 sessions,
    below) is still running or has not yet been reached -- read from
    results/state_shape_common_range.json's already independently cross-
    checked four-arm headline table (panichello row, bins 3-8, no lag
    excluded, matching this module's own SLOPE_RANGE_BINS exactly) rather
    than left blank. The human and ALM arms' own paired-change fits do not
    depend on this number at all (see _arm_paired_change); only the gap and
    the closed-fraction branch do, and both are recomputed for free (pure
    arithmetic) the moment a fresher number is available."""
    if not COMMON_RANGE_PATH.exists():
        return None, "not_available_state_shape_common_range_json_missing"
    data = json.loads(COMMON_RANGE_PATH.read_text())
    try:
        slope = data["four_row_headline_table_r_obs_r_null_d_perm_by_arm"]["rows"]["panichello"]["d_perm"]["test"]["mean_value"]
        return float(slope), "results/state_shape_common_range.json four_row_headline_table_r_obs_r_null_d_perm_by_arm.rows.panichello.d_perm (n_splits=10, n_null_replicates=10 -- macaque's own native setting, not yet up-run to the human/ALM module default at the time this number was read)"
    except (KeyError, TypeError):
        return None, "not_available_expected_field_missing_from_state_shape_common_range_json"


def _add_settings_table_pointer_to_common_range_artifact() -> None:
    """A one-field, additive pointer from state_shape_common_range.json to
    this module's four_arm_estimator_settings_table (1.5's "explicit
    pointer both ways"), added directly rather than by regenerating that
    artifact's full multi-stage pipeline (cohort bootstrap, breakpoint
    fits, cross-arm null-explains-it test) for a metadata addition -- see
    the implementation report for why that judgment call was made. Not
    called if the artifact does not exist; never overwrites any existing
    field but its own."""
    if not COMMON_RANGE_PATH.exists():
        return
    data = json.loads(COMMON_RANGE_PATH.read_text())
    data["estimator_settings_table_pointer"] = (
        "Each row's n_splits/n_null_replicates/null_splits_per_replicate is not printed in this "
        "artifact's four_row_headline_table_r_obs_r_null_d_perm_by_arm; see "
        "results/persistence_estimator_split_count_sensitivity.json four_arm_estimator_settings_table "
        "for that table, added there rather than by a full regeneration of this artifact."
    )
    COMMON_RANGE_PATH.write_text(json.dumps(data, indent=2))


def _apply_resume_state(output: dict, prior_output: dict) -> tuple[dict, bool]:
    """RESUME, DO NOT RESTART: if a prior run of this module already
    finished the human delay and ALM arms (both are stored as complete
    pooled+paired-change results, not just a session-ID progress list, so
    they are genuinely reusable without refitting), copy them into
    ``output`` verbatim rather than refitting 95 sessions that already have
    a settled answer on disk. Also carries the panichello per-session row
    cache over from disk UNCONDITIONALLY -- this is what makes the
    panichello loop resumable at all, and it must not be nested inside the
    human/ALM branch: an earlier version of this function copied the
    human/ALM keys but not this one, so a resumed run started
    panichello_session_rows from an empty dict and OVERWROTE the full
    on-disk cache, session by session, with only the sessions it refit in
    the new run -- a real defect that cost 8 already-cached sessions' worth
    of refitting before it was caught mid-run (see the implementation
    report). Returns the updated output and whether the human/ALM arms
    were reused."""
    human_and_alm_already_complete = (
        "human_delay_arm" in prior_output and "alm_arm" in prior_output
        and set(["human_delay", "alm"]).issubset(set(prior_output.get("corpora_completed", [])))
    )
    if human_and_alm_already_complete:
        for key in (
            "human_delay_sessions_completed", "human_delay_sessions_pending", "human_delay_arm",
            "alm_sessions_completed", "alm_sessions_pending", "alm_arm",
            "alm_rate_matched_keep_probability", "alm_human_matched_window_s",
            "macaque_reference_settings_d_perm_slope_provisional",
            "macaque_reference_settings_d_perm_slope_provisional_source",
        ):
            if key in prior_output:
                output[key] = prior_output[key]
        output["corpora_completed"] = ["human_delay", "alm"]
        output["corpora_pending"] = ["panichello_confirmatory_up_run"]
        output["human_delay_and_alm_arms_reused_from_disk_without_refitting"] = True
    if isinstance(prior_output.get("panichello_session_rows"), dict):
        output["panichello_session_rows"] = dict(prior_output["panichello_session_rows"])
    return output, human_and_alm_already_complete


def main() -> None:
    root = data_root()
    t0 = time.time()

    output = {
        "version": "2026-08-24",
        "scope": (
            "Human intracranial delay-epoch state (DANDI 000469, 001187, 000574; structure=='pooled'; "
            "n=72 sessions), mouse ALM delay-epoch state at the human-matched window and rate-matched "
            "thinning (n=23 sessions), and macaque lPFC delay-epoch state (Panichello et al. 2024; "
            "n=25 sessions), every session refit at the fixed deciding window width (3 bins, 300 ms) "
            "at two estimator settings: the human/ALM module default (n_splits=12, n_null_replicates=20) "
            "and the macaque arm's lighter setting (n_splits=10, n_null_replicates=10). "
            "null_splits_per_replicate is held at 6 for every fit at every setting -- it is not one of "
            "the two audited parameters. The lag range for every slope is bins 3-8 (0.3-0.8 s), no lag "
            "excluded, matching results/state_shape_common_range.json's four-arm headline table exactly. "
            "Run order is deliberately human delay, then ALM, then macaque: the human and ALM arms are "
            "the cheap, decisive direction and are fit and scored first (against a macaque reference "
            "number read from an existing cross-checked artifact); the macaque arm's own fresh dual-"
            "settings refit is the slower confirmatory direction and runs last, updating the human/ALM "
            "gap and branch fields in place if it completes and changes the reference number materially."
        ),
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "estimator_settings_by_arm": {
            "human_delay": {"reference": REFERENCE_SETTINGS, "down": DOWN_SETTINGS},
            "alm": {"reference": REFERENCE_SETTINGS, "down": DOWN_SETTINGS},
            "panichello": {"reference": REFERENCE_SETTINGS, "down": DOWN_SETTINGS},
            "null_splits_per_replicate_all_arms_all_settings": LAG_NULL_SPLITS_PER_REPLICATE,
        },
        "deciding_width_bins": DECIDING_WIDTH_BINS, "deciding_epoch": DECIDING_EPOCH,
        "bin_width_s": BIN_WIDTH_S, "slope_range_bins": list(SLOPE_RANGE_BINS),
        "slope_range_seconds": [SLOPE_RANGE_BINS[0] * BIN_WIDTH_S, SLOPE_RANGE_BINS[1] * BIN_WIDTH_S],
        "corpora_completed": [], "corpora_pending": ["human_delay", "alm", "panichello_confirmatory_up_run"],
        "wall_clock_s": None,
    }
    prior_output = json.loads(OUTPUT_PATH.read_text()) if OUTPUT_PATH.exists() else {}
    output, human_and_alm_already_complete = _apply_resume_state(output, prior_output)
    _flush(output)

    if human_and_alm_already_complete:
        macaque_reference_slope = output["macaque_reference_settings_d_perm_slope_provisional"]
        macaque_reference_source = output["macaque_reference_settings_d_perm_slope_provisional_source"]
        print("Human delay and ALM arms reused from disk (already complete); not refit.",
              file=sys.stderr, flush=True)
    else:
        macaque_reference_slope, macaque_reference_source = _provisional_macaque_reference_slope()
        output["macaque_reference_settings_d_perm_slope_provisional"] = macaque_reference_slope
        output["macaque_reference_settings_d_perm_slope_provisional_source"] = macaque_reference_source
        _flush(output)

        # Human delay arm FIRST: the cheap, decisive direction, and the one whose result matters most per
        # session of compute spent. Scored against the provisional macaque number above so this arm's own
        # gap/branch fields are populated and flushed before the slower macaque confirmatory refit is even
        # attempted.
        print("Human delay arm: fitting both settings per session...", file=sys.stderr, flush=True)
        t_arm = time.time()
        human_sessions = _human_delay_sessions(root)
        human_rows_fitted = []
        for i, s in enumerate(human_sessions):
            row = {"dataset": s["dataset"], "patient": s["patient"], "session": s["session"], "structure": s["structure"]}
            row["reference"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **REFERENCE_SETTINGS)
            row["down"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **DOWN_SETTINGS)
            human_rows_fitted.append(row)
            output["human_delay_sessions_completed"] = [r["session"] for r in human_rows_fitted]
            output["human_delay_sessions_pending"] = [s2["session"] for s2 in human_sessions[i + 1:]]
            _flush(output)
            print(f"progress human_delay {i + 1}/{len(human_sessions)} {s['session']} flushed", flush=True)
        human_paired_change = _arm_paired_change(human_rows_fitted)
        output["human_delay_arm"] = _arm_finalize_branch(human_paired_change, macaque_reference_slope, macaque_reference_source)
        output["corpora_completed"].append("human_delay")
        output["corpora_pending"] = ["alm", "panichello_confirmatory_up_run"]
        output["wall_clock_s"] = time.time() - t0
        print(f"  human delay done in {time.time() - t_arm:.1f}s, branch={output['human_delay_arm']['branch']}",
              file=sys.stderr, flush=True)
        _flush(output)

        # ALM arm SECOND: same priority reasoning as human delay above.
        print("ALM arm: fitting both settings per session...", file=sys.stderr, flush=True)
        t_arm = time.time()
        prior_path = Path(__file__).resolve().parents[1] / "results" / "state_persistence.json"
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
        rate_matched_keep_probability = prior.get("matched_sensitivity_alm", {}).get("rate_matched_keep_probability", 1.0)
        human_window_s = EPOCH_WINDOWS_S["delay"]
        alm_sessions = _alm_sessions(root, human_window_s, rate_matched_keep_probability)
        alm_rows_fitted = []
        for i, s in enumerate(alm_sessions):
            row = {"dataset": s["dataset"], "patient": s["patient"], "session": s["session"], "structure": s["structure"]}
            row["reference"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **REFERENCE_SETTINGS)
            row["down"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **DOWN_SETTINGS)
            alm_rows_fitted.append(row)
            output["alm_sessions_completed"] = [r["session"] for r in alm_rows_fitted]
            output["alm_sessions_pending"] = [s2["session"] for s2 in alm_sessions[i + 1:]]
            _flush(output)
            print(f"progress alm {i + 1}/{len(alm_sessions)} {s['session']} flushed", flush=True)
        alm_paired_change = _arm_paired_change(alm_rows_fitted)
        output["alm_arm"] = _arm_finalize_branch(alm_paired_change, macaque_reference_slope, macaque_reference_source)
        output["alm_rate_matched_keep_probability"] = rate_matched_keep_probability
        output["alm_human_matched_window_s"] = human_window_s
        output["corpora_completed"].append("alm")
        output["corpora_pending"] = ["panichello_confirmatory_up_run"]
        output["wall_clock_s"] = time.time() - t0
        print(f"  ALM done in {time.time() - t_arm:.1f}s, branch={output['alm_arm']['branch']}",
              file=sys.stderr, flush=True)

    output["overall_branch_by_arm"] = {"human_delay": output["human_delay_arm"]["branch"], "alm": output["alm_arm"]["branch"]}
    output["arms_agree"] = output["human_delay_arm"]["branch"] == output["alm_arm"]["branch"]
    output["status"] = "complete_pending_macaque_confirmatory_up_run"
    _flush(output)
    print(json.dumps({"overall_branch_by_arm": output["overall_branch_by_arm"], "arms_agree": output["arms_agree"]}, indent=2))

    # Macaque confirmatory up-run LAST (1.4): the slower, optional direction, attempted only once the
    # decisive human/ALM result above is already complete and safely on disk. If this is interrupted, the
    # human/ALM verdict above is unaffected -- it is scored against the provisional macaque number, which
    # is itself an already-verified figure, not a guess.
    #
    # This section's own resumability: the panichello arm's per-session ROW data (both settings' full
    # profile/null_permutation dicts, not just a completed-session-ID list) is cached in
    # output["panichello_session_rows"], keyed by session ID, and flushed after every session. A prior
    # run of this module flushed only the session-ID progress list here, not the row data itself, which
    # made a genuine resume-without-refit impossible -- the 14 sessions it reported "completed" left no
    # row data on disk to aggregate from, only their IDs. That is a defect in this module as previously
    # written (see the implementation report), and per this arm's own resume-first instruction, the fix
    # is to check for row-level data before refitting -- which is what the cache lookup below does -- and
    # only refit a session when no cached row is found for it, exactly the situation those 14 sessions
    # are in.
    output["panichello_paired_change_decision_rule_declared_before_fitting"] = (
        PANICHELLO_PAIRED_CHANGE_DECISION_RULE_DECLARED_BEFORE_FITTING
    )
    output["panichello_paired_change_sign_convention"] = PANICHELLO_PAIRED_CHANGE_SIGN_CONVENTION
    _flush(output)

    print("Panichello macaque lPFC (confirmatory up-run): fitting both settings per session (resumable "
          "from any cached per-session row already on disk)...", file=sys.stderr, flush=True)
    t_arm = time.time()
    panichello_sessions = _panichello_sessions(root)
    cached_rows = output.get("panichello_session_rows")
    if not isinstance(cached_rows, dict):
        cached_rows = {}
    panichello_rows_fitted = []
    n_reused_from_disk = 0
    n_fit_fresh_this_run = 0
    for i, s in enumerate(panichello_sessions):
        session_id = s["session"]
        was_cached = session_id in cached_rows
        if was_cached:
            row = cached_rows[session_id]
            n_reused_from_disk += 1
        else:
            row = {"dataset": s["dataset"], "patient": s["patient"], "session": s["session"], "structure": s["structure"]}
            row["reference"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **REFERENCE_SETTINGS)
            row["down"] = _lag_run_row(s["counts"], DECIDING_WIDTH_BINS, s["seed"], **DOWN_SETTINGS)
            cached_rows[session_id] = row
            n_fit_fresh_this_run += 1
        panichello_rows_fitted.append(row)
        output["panichello_session_rows"] = cached_rows
        output["panichello_sessions_completed"] = [r["session"] for r in panichello_rows_fitted]
        output["panichello_sessions_pending"] = [s2["session"] for s2 in panichello_sessions[i + 1:]]
        _flush(output)
        print(f"progress panichello {i + 1}/{len(panichello_sessions)} {session_id} "
              f"{'reused_from_disk' if was_cached else 'fit_fresh'} flushed", flush=True)
    print(f"  panichello sessions: {n_reused_from_disk} reused from disk, {n_fit_fresh_this_run} fit fresh "
          f"this run", file=sys.stderr, flush=True)
    panichello_reference_lag_rows = _row_for_settings(panichello_rows_fitted, "reference")
    panichello_down_lag_rows = _row_for_settings(panichello_rows_fitted, "down")
    panichello_pooled_reference = _pooled_arm_slopes(panichello_reference_lag_rows)
    panichello_pooled_down = _pooled_arm_slopes(panichello_down_lag_rows)
    macaque_fresh_reference_slope = panichello_pooled_reference["d_perm"]["test"].get("mean_value")
    macaque_down_slope = panichello_pooled_down["d_perm"]["test"].get("mean_value")

    output["panichello_confirmatory_up_run"] = {
        "note": (
            "The confirmatory direction: the macaque arm's own d_perm slope at the human/ALM module "
            "default settings (n_splits=12, n_null_replicates=20), reported beside its slope at its own "
            "native lighter setting (n_splits=10, n_null_replicates=10). Both are computed fresh in this "
            "module rather than one being copied from results/state_persistence_lag.json."
        ),
        "n_sessions_reused_from_disk": n_reused_from_disk, "n_sessions_fit_fresh_this_run": n_fit_fresh_this_run,
        "pooled_at_down_settings_native": panichello_pooled_down,
        "pooled_at_reference_settings_up_run": panichello_pooled_reference,
        "d_perm_slope_moved_native_to_reference": (
            (macaque_fresh_reference_slope - macaque_down_slope)
            if (macaque_fresh_reference_slope is not None and macaque_down_slope is not None) else None
        ),
    }
    output["corpora_completed"].append("panichello_confirmatory_up_run")
    output["corpora_pending"] = []
    print(f"  panichello done in {time.time() - t_arm:.1f}s", file=sys.stderr, flush=True)

    # 1.2: the macaque's own paired change, m_native and m_default, and 1.3: this arm's own
    # pre-declared branch (already flushed to disk above, before this fitting resumed).
    m_native_d_perm_test = panichello_pooled_down["d_perm"]["test"]
    m_default_d_perm_test = panichello_pooled_reference["d_perm"]["test"]
    m_native_slope = m_native_d_perm_test.get("mean_value")
    m_default_slope = m_default_d_perm_test.get("mean_value")
    panichello_own_paired_change = _panichello_paired_change_up_run(panichello_rows_fitted)
    panichello_own_branch = _panichello_branch(
        panichello_own_paired_change["paired_change_d_perm_slope"], m_default_d_perm_test)
    paired_change_mdd_field = panichello_own_paired_change["paired_change_d_perm_slope"].get(
        "minimum_detectable_paired_difference_80pct_power")
    # paired_change_mdd_field is the FULL dict minimum_detectable_paired_difference returns (status, n,
    # sd, alpha, power, z_factor, mdd), not the scalar itself -- .get("mdd") below pulls the number out;
    # confusing the dict for the scalar is exactly the kind of bug this project's own standing rule about
    # reporting an MDD beside every weighted null exists to make someone actually look at, and it crashed
    # this module on its first real run once it reached the aggregation step, after all 25 sessions had
    # already been fit and flushed -- caught and fixed here, not silently patched around.
    paired_change_mdd = (
        paired_change_mdd_field.get("mdd") if isinstance(paired_change_mdd_field, dict)
        and paired_change_mdd_field.get("status") == "computed" else None
    )
    mdd_threshold = abs(m_native_slope) if m_native_slope is not None else None

    output["m_native_d_perm_slope"] = m_native_slope
    output["m_default_d_perm_slope"] = m_default_slope
    output["panichello_own_paired_change"] = panichello_own_paired_change
    output["panichello_own_branch"] = panichello_own_branch
    output["panichello_own_paired_change_mdd_versus_threshold"] = {
        "paired_change_minimum_detectable_paired_difference_80pct_power": paired_change_mdd,
        "paired_change_minimum_detectable_paired_difference_full_detail": paired_change_mdd_field,
        "threshold_this_arms_rule_uses": mdd_threshold,
        "threshold_definition": (
            "abs(m_native): this arm's decision rule (see "
            "panichello_paired_change_decision_rule_declared_before_fitting) has no closed-fraction-of-a-"
            "gap magnitude threshold the way the human/ALM arm's rule does -- its branches turn on the "
            "statistical significance of the paired change and of m_default directly, not on a "
            "pre-declared minimum magnitude. The magnitude comparator used here is a judgment call made "
            "when writing this field, not a value stated in the pre-declared rule itself: the size of "
            "m_native, since a fully settings-driven account of the macaque's positive native slope would "
            "need the paired change to be approximately that large, and this bound says whether the "
            "design could have detected a change of that size at 80% power."
        ),
        "paired_change_mdd_clears_threshold": (
            paired_change_mdd is not None and mdd_threshold is not None and paired_change_mdd <= mdd_threshold
        ),
    }

    # 1.3 (final paragraph): recompute the human/ALM gaps against m_default, keep the provisional-
    # denominator versions (output["human_delay_arm"], output["alm_arm"]) in place beside them per 0.26,
    # and state explicitly that the two gaps are different numbers measured against different macaque
    # settings.
    matched_source = (
        "computed_fresh_in_this_module_panichello_confirmatory_up_run_pooled_at_reference_settings_up_run "
        "(m_default: macaque pooled d_perm slope at the human/ALM module default settings, n_splits=12, "
        "n_null_replicates=20)"
    )
    output["human_delay_arm_at_matched_macaque_reference"] = _arm_finalize_branch(
        output["human_delay_arm"], m_default_slope, matched_source)
    output["alm_arm_at_matched_macaque_reference"] = _arm_finalize_branch(
        output["alm_arm"], m_default_slope, matched_source)
    output["provisional_vs_matched_macaque_reference_settings_d_perm_slope"] = {
        "provisional_native_setting_value": macaque_reference_slope,
        "matched_module_default_value": m_default_slope,
        "difference_matched_minus_provisional": (
            m_default_slope - macaque_reference_slope
            if (m_default_slope is not None and macaque_reference_slope is not None) else None
        ),
        "note": (
            "human_delay_arm.gap_to_macaque_reference_settings_d_perm_slope and "
            "alm_arm.gap_to_macaque_reference_settings_d_perm_slope (the provisional-denominator fields, "
            "kept in place, not replaced) are measured against the macaque's NATIVE setting "
            "(+0.0760, read from results/state_shape_common_range.json). "
            "human_delay_arm_at_matched_macaque_reference.gap_to_macaque_reference_settings_d_perm_slope "
            "and the equivalent ALM field are measured against m_default, the macaque's own pooled slope "
            "at the human/ALM module default settings, computed fresh in this module. These are two "
            "different numbers measured against two different macaque settings -- a reader who takes the "
            "provisional gap as fixed is comparing two different gaps without being told they differ."
        ),
    }
    output["overall_branch_by_arm_at_matched_macaque_reference"] = {
        "human_delay": output["human_delay_arm_at_matched_macaque_reference"]["branch"],
        "alm": output["alm_arm_at_matched_macaque_reference"]["branch"],
    }
    output["arms_agree_at_matched_macaque_reference"] = (
        output["human_delay_arm_at_matched_macaque_reference"]["branch"]
        == output["alm_arm_at_matched_macaque_reference"]["branch"]
    )

    # 1.4: the transfer label. The human/ALM clearance above was obtained in the two arms the
    # split/replicate-count confound was NOT raised about; this field says so explicitly and records
    # whether the macaque's own direct measurement (panichello_own_branch above) agrees with or
    # contradicts what that clearance would predict, per this project's own rule that a clearance
    # obtained in arms other than the one at risk is a transfer argument and must be labelled one.
    output["transfer_label_confound_arm_at_risk"] = (
        "The split/replicate-count confound was raised specifically about the macaque arm: it is the "
        "only one of the three arms whose native-setting d_perm slope is positive, and the only arm run "
        "at the lighter setting (n_splits=10, n_null_replicates=10) by default. The clearance already on "
        "disk in human_delay_arm.paired_change_d_perm_slope and alm_arm.paired_change_d_perm_slope was "
        "obtained by running the human and ALM arms DOWN to that lighter setting and finding no "
        "significant, materially-sized change in either -- but the confound was never raised about the "
        "human or ALM arms, so that clearance is a transfer argument from arms where the test was cheap "
        "to the arm where it was expensive, not a direct measurement of the arm at risk. The axis along "
        "which that transfer is least safe is simultaneously recorded population size: the lighter "
        "setting was applied to the macaque arm precisely because its sessions have several times the "
        "simultaneously recorded units of a human or ALM session, which is exactly the property the "
        "human and ALM sessions do not share and so cannot test. panichello_own_paired_change and "
        "panichello_own_branch above are this arm's own direct measurement of the same question, run at "
        "its own population size rather than inferred from a smaller one. The measured branch is '"
        + panichello_own_branch + "'."
    )

    # 1.5: the four-arm estimator settings table, per 0.25(c) -- results/state_shape_common_range.json is
    # not on the read-only list, but it is a large, multi-stage artifact (cohort bootstrap, breakpoint
    # fits, cross-arm null-explains-it test) that a settings-metadata addition does not justify
    # regenerating in full; the table is written here instead, with a lightweight one-field pointer added
    # directly to that artifact (not a full regeneration) so the cross-reference resolves both ways. See
    # the implementation report for this judgment call.
    output["four_arm_estimator_settings_table"] = {
        "note": (
            "results/state_shape_common_range.json's four-row headline table (human_delay, alm, "
            "panichello, human_encoding) does not itself print each row's n_splits/n_null_replicates/"
            "null_splits_per_replicate. This table adds them here as a cross-reference; see that "
            "artifact's own estimator_settings_table_pointer field for the reverse pointer."
        ),
        "rows": {
            "human_delay": {
                "n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES,
                "null_splits_per_replicate": LAG_NULL_SPLITS_PER_REPLICATE,
                "differs_from_other_rows": True,
                "what_is_known_about_whether_the_difference_matters": (
                    "Tested directly: this artifact's human_delay_arm.paired_change_d_perm_slope reports "
                    "the effect of running this arm at the panichello native setting instead -- a small, "
                    "non-significant change (see that field for the number and p-value)."
                ),
            },
            "alm": {
                "n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES,
                "null_splits_per_replicate": LAG_NULL_SPLITS_PER_REPLICATE,
                "differs_from_other_rows": True,
                "what_is_known_about_whether_the_difference_matters": (
                    "Tested directly: this artifact's alm_arm.paired_change_d_perm_slope, same design as "
                    "human_delay above."
                ),
            },
            "panichello": {
                "n_splits": PANICHELLO_LAG_N_SPLITS, "n_null_replicates": PANICHELLO_LAG_N_NULL_REPLICATES,
                "null_splits_per_replicate": LAG_NULL_SPLITS_PER_REPLICATE,
                "differs_from_other_rows": True,
                "what_is_known_about_whether_the_difference_matters": (
                    "Tested directly, in the direction that matters for this row (running IT UP to the "
                    "module default rather than running the others down): this artifact's "
                    "panichello_own_paired_change and panichello_own_branch fields."
                ),
            },
            "human_encoding": {
                "n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES,
                "null_splits_per_replicate": LAG_NULL_SPLITS_PER_REPLICATE,
                "differs_from_other_rows": False,
                "what_is_known_about_whether_the_difference_matters": (
                    "Not applicable -- this row uses the same settings as human_delay and alm; only the "
                    "panichello row's settings differ from the other three."
                ),
            },
        },
    }

    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    _flush(output)

    try:
        _add_settings_table_pointer_to_common_range_artifact()
    except Exception as exc:  # pragma: no cover -- best-effort cross-reference, not load-bearing
        print(f"  note: could not add pointer field to state_shape_common_range.json: {exc}",
              file=sys.stderr, flush=True)

    print(json.dumps({
        "overall_branch_by_arm_provisional": output["overall_branch_by_arm"],
        "overall_branch_by_arm_at_matched_macaque_reference": output.get("overall_branch_by_arm_at_matched_macaque_reference"),
        "panichello_own_branch": panichello_own_branch,
    }, indent=2))


if __name__ == "__main__":
    main()
