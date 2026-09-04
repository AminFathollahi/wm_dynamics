#!/usr/bin/env python3
"""run_band_versus_sensor_decomposition.py -- separates two entangled
explanations for why every non-invasive arm of this project's observability
census has died: a frequency-content limit (the feature this project's
intracranial arms use, high-gamma power, is simply not present in a
non-invasive recording's representable band) and a sensor limit (electrode
count, placement, distance and volume conduction, independent of band).

The two are entangled by physics, not by choice: DANDI 000574's scalp-EEG
series (ecephys.eeg, a 19-channel 10-20 montage) is sampled at roughly
139.8 Hz in the same NWB file, the same patients, the same trials and the
same recording as its depth-contact series -- but its Nyquist frequency,
roughly 69.9 Hz, sits entirely below this project's 70-150 Hz high-gamma
band. Asking that series for high-gamma power is not attenuated or noisy; it
is not representable at all, and src/preprocessing.py's bandpass_filter now
raises a named error for exactly this request instead of returning a
coerced number.

So this script does not compare grains on one feature. It fixes a low band
(8-45 Hz, entirely below both the 50 Hz mains notch and the scalp series'
own Nyquist) and runs three of the four cells of a two-by-two:

    depth contact, high gamma (70-150 Hz)  ... read from the existing
                                                census and factor-analysis
                                                comparison, never refit
    depth contact, low band (8-45 Hz)      ... NEW, this script
    scalp EEG,     low band (8-45 Hz)      ... NEW, staged into the census
                                                by scripts/run_observability_
                                                and_power_census.py, reused
                                                here
    scalp EEG,     high gamma              ... physically impossible, not
                                                attempted

Two comparisons fall out, each holding one thing fixed:

    Comparison A (band effect, fixed sensor): depth at high gamma versus
    depth at the low band, same sensor, same contacts, same trials.

    Comparison B (sensor effect, fixed band): depth at the low band versus
    scalp at the low band, same band, same patients, same trials, matched
    to the scalp grain's own recording validity.

The deciding quantity for both comparisons is the factor-analysis model's
own observation-noise variance fraction (src/observability.py's estimator
family is a different, complementary estimator used by the census; this one
matches scripts/run_latent_model_observation_noise_comparison.py's
dimensionality_and_noise_term exactly, at the same matched latent
dimensionality). The persistence contrast's level and the participation
ratio in both bases are reported beside it, never as tie-breakers.

Deliverable: results/band_versus_sensor_decomposition.json.

Resumability: every session's three tensor variants (depth at the low band
matched to Comparison A's own trial set, depth at the low band matched to
Comparison B's trial set, scalp at the low band) at both bin widths are
flushed to results/.checkpoints/band_versus_sensor_decomposition_checkpoint.json
immediately after that session completes, keyed by session, so a kill costs
one session's work, not the run.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus_sessions import data_root  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from preprocessing import load_boran_nwb  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from run_boran_modality_consistency import (  # noqa: E402
    BAD_CHANNEL_MAD_THRESHOLD, BORAN_MAINS_HZ, MAINT_WIN, MIN_BIPOLAR_CHANNELS,
    MIN_TRIALS, lfp_maintenance_tensor, registry_sessions, scalp_low_band_maintenance_tensor,
)
from run_latent_model_observation_noise_comparison import analyze_session  # noqa: E402
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402

SEED = 20260812
LOW_BAND_LO_HZ = 8.0
LOW_BAND_HI_HZ = 45.0

CENSUS_PATH = ROOT / "results" / "observability_and_power_census.json"
FACTOR_MODEL_ARTIFACT_PATH = ROOT / "results" / "latent_model_observation_noise_comparison.json"
FACTOR_MODEL_CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "latent_model_observation_noise_comparison_checkpoint.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "band_versus_sensor_decomposition_checkpoint.json"
OUTPUT_PATH = ROOT / "results" / "band_versus_sensor_decomposition.json"

BIN_WIDTHS_MS = (100, 200)


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Read the precursors. Neither is refit.
# ---------------------------------------------------------------------------

def load_existing_high_gamma_sessions() -> dict[tuple[str, str, int], dict]:
    """dandi_000574 depth-contact-LFP per-session factor-analysis results at
    the project's standard high-gamma band, read from the already-complete
    checkpoint block scripts/run_latent_model_observation_noise_comparison.py
    wrote -- never refit here. Keyed by (patient, session, bin_ms)."""
    checkpoint = json.loads(FACTOR_MODEL_CHECKPOINT_PATH.read_text())
    block = checkpoint["dandi_000574_depth_contact_lfp"]
    if block["status"] != "complete":
        raise ValueError(
            "results/.checkpoints/latent_model_observation_noise_comparison_checkpoint.json's "
            "dandi_000574_depth_contact_lfp block is not marked complete -- the existing high-gamma "
            "comparison this script reads must be finished and on disk before Comparison A can be paired "
            "against it."
        )
    return {(s["patient"], s["session"], s["bin_ms"]): s for s in block["sessions"]}


def reproducibility_check_against_existing_checkpoint(existing_sessions: dict) -> dict:
    """Guard against silently reusing a checkpoint that current code would no
    longer reproduce, for the one checkpoint this script consumes without
    refitting: pick one cached
    dandi_000574_depth_contact_lfp session, re-fit it with CURRENT code
    (lfp_maintenance_tensor's now-parameterised lo/hi, called with its
    unchanged high-gamma defaults) and diff the returned fields against the
    cached copy. Run once, before any cached entry is relied on for pairing."""
    key = min(existing_sessions.keys())
    cached = existing_sessions[key]
    patient, session, bin_ms = key
    reg_row = next(r for r in registry_sessions(data_root()) if r["session"] == session)
    path = data_root() / reg_row["path"]
    with h5py.File(path, "r") as handle:
        trials = handle["intervals/trials"]
        artifact = trials["artifact"][:].astype(bool)
    ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True,
                           bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ)
    keep = (~artifact) & ieeg["valid"]
    n_bins = int(round(MAINT_WIN / (bin_ms / 1000.0)))
    tensor = lfp_maintenance_tensor(ieeg, keep, n_bins=n_bins)  # default lo/hi: unchanged high-gamma band
    fresh = analyze_session("dandi_000574", "depth_contact_lfp", "depth_contact_lfp", patient, session,
                             tensor.astype(float), bin_ms, is_point_process=False)
    fields_to_check = [
        ("dimensionality", "pca", "participation_ratio"),
        ("dimensionality", "factor_analysis", "observation_noise_variance_fraction"),
        ("dimensionality", "factor_analysis", "participation_ratio"),
        ("persistence_contrast", "pca", "level"),
        ("persistence_contrast", "factor_analysis", "level"),
    ]

    def _get(d, path_):
        for p in path_:
            if d is None or p not in d:
                return None
            d = d[p]
        return d

    diffs = {}
    for path_ in fields_to_check:
        cached_v = _get(cached, path_)
        fresh_v = _get(fresh, path_)
        name = ".".join(path_)
        if cached_v != fresh_v:
            diffs[name] = {"cached": cached_v, "fresh": fresh_v}

    return {
        "loader_path_checked": "dandi_000574_depth_contact_lfp (registry_sessions + lfp_maintenance_tensor)",
        "session_checked": f"{patient}/{session}/bin{bin_ms}",
        "fields_checked": [".".join(p) for p in fields_to_check],
        "diff": diffs,
        "diff_is_empty": len(diffs) == 0,
        "n_trials_cached": cached.get("n_trials"), "n_trials_fresh": fresh.get("n_trials"),
        "n_units_cached": cached.get("n_units"), "n_units_fresh": fresh.get("n_units"),
    }


# ---------------------------------------------------------------------------
# Per-session computation. Checkpointed per session (rule: flush
# incrementally, resumable without refitting).
# ---------------------------------------------------------------------------

def _degeneracy_reason(tensor: np.ndarray) -> str | None:
    """None if the tensor has real variance to fit; a stated reason if every
    channel is exactly flat. A completely flat recording (verified directly:
    dandi_000574 sub-08_ses-05's raw ecephys.eeg series is exactly zero on
    every sample, every channel -- a dead acquisition channel, not noise) is
    not a hard-to-fit case, it is not data. src/observability.py's nugget-
    fraction estimator already declines gracefully on it (zero positive
    autocovariance lags, reported as fewer_than_three_positive_lags); the
    factor-analysis EM solver this script also uses has no such graceful
    floor and can spend many iterations dividing by a zero variance before
    giving up, which is disproportionate wall-clock for a cell that was
    never going to produce a real measurement. Caught here, before the
    expensive fit, rather than after."""
    if not np.any(tensor.std(axis=(0, 2)) > 0.0):
        return "every channel is exactly flat (zero variance) in this tensor -- a dead recording, not low-amplitude data"
    return None


def _process_session(reg_row: dict) -> dict:
    session_key = reg_row["session"]
    patient = session_key.split("_ses-")[0]
    path = data_root() / reg_row["path"]
    record: dict = {"patient": patient, "session": session_key}

    if not path.is_file():
        record["funnel_stage_reached"] = "registered"
        record["reason_stopped"] = f"registry path not found on disk: {path}"
        return record

    try:
        with h5py.File(path, "r") as handle:
            trials = handle["intervals/trials"]
            artifact = trials["artifact"][:].astype(bool)
        ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True,
                               bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ)
    except Exception as exc:  # noqa: BLE001
        record["funnel_stage_reached"] = "registered"
        record["reason_stopped"] = f"depth-contact series did not load: {type(exc).__name__}: {exc}"
        return record

    if ieeg["epochs"].shape[0] != len(artifact):
        record["funnel_stage_reached"] = "registered"
        record["reason_stopped"] = "depth-contact trial count disagrees with the trial table"
        return record

    keep_a = (~artifact) & ieeg["valid"]  # Comparison A's own trial set: matches the existing high-gamma cell exactly
    n_depth_channels = ieeg["epochs"].shape[1]
    if int(keep_a.sum()) < MIN_TRIALS:
        record["funnel_stage_reached"] = "registered"
        record["reason_stopped"] = f"only {int(keep_a.sum())} depth-contact trials pass artifact+window validity"
        return record

    record["funnel_stage_reached"] = "depth_contact_survives"
    record["n_trials_comparison_a"] = int(keep_a.sum())

    try:
        eeg = load_boran_nwb(str(path), signal="eeg")
    except Exception as exc:  # noqa: BLE001
        record["reason_stopped"] = f"scalp series did not load: {type(exc).__name__}: {exc}"
        return record

    if eeg["epochs"].shape[0] != len(artifact):
        record["reason_stopped"] = "scalp trial count disagrees with the trial table"
        return record

    keep_b = keep_a & eeg["valid"]  # Comparison B's own trial set: matched between depth and scalp
    if int(keep_b.sum()) < MIN_TRIALS:
        record["reason_stopped"] = f"only {int(keep_b.sum())} trials pass artifact+window validity matched to the scalp grain"
        return record

    n_scalp_channels_before_mastoid = len(eeg["electrode_labels"])

    record["funnel_stage_reached"] = "depth_and_scalp_both_survive"
    record["n_trials_comparison_b"] = int(keep_b.sum())
    record["n_depth_channels"] = int(n_depth_channels)
    record["n_scalp_channels_before_mastoid_exclusion"] = int(n_scalp_channels_before_mastoid)

    cells: dict[str, dict] = {}
    skipped_cells: dict[str, str] = {}
    for bin_ms in BIN_WIDTHS_MS:
        n_bins = int(round(MAINT_WIN / (bin_ms / 1000.0)))

        depth_a_tensor = lfp_maintenance_tensor(ieeg, keep_a, n_bins=n_bins, lo=LOW_BAND_LO_HZ, hi=LOW_BAND_HI_HZ)
        key = f"depth_low_band_comparison_a_bin{bin_ms}"
        if depth_a_tensor.shape[1] < MIN_BIPOLAR_CHANNELS:
            skipped_cells[key] = f"only {depth_a_tensor.shape[1]} bipolar channels after QC"
        elif _degeneracy_reason(depth_a_tensor) is not None:
            skipped_cells[key] = _degeneracy_reason(depth_a_tensor)
        else:
            cells[key] = analyze_session("dandi_000574", "depth_contact_lfp", "depth_contact_lfp", patient,
                                          session_key, depth_a_tensor.astype(float), bin_ms, is_point_process=False)

        depth_b_tensor = lfp_maintenance_tensor(ieeg, keep_b, n_bins=n_bins, lo=LOW_BAND_LO_HZ, hi=LOW_BAND_HI_HZ)
        key = f"depth_low_band_comparison_b_bin{bin_ms}"
        if depth_b_tensor.shape[1] < MIN_BIPOLAR_CHANNELS:
            skipped_cells[key] = f"only {depth_b_tensor.shape[1]} bipolar channels after QC"
        elif _degeneracy_reason(depth_b_tensor) is not None:
            skipped_cells[key] = _degeneracy_reason(depth_b_tensor)
        else:
            cells[key] = analyze_session("dandi_000574", "depth_contact_lfp", "depth_contact_lfp", patient,
                                          session_key, depth_b_tensor.astype(float), bin_ms, is_point_process=False)

        scalp_tensor, kept_labels = scalp_low_band_maintenance_tensor(
            eeg, keep_b, n_bins=n_bins, lo=LOW_BAND_LO_HZ, hi=LOW_BAND_HI_HZ)
        key = f"scalp_low_band_bin{bin_ms}"
        record[f"n_scalp_channels_after_mastoid_exclusion_bin{bin_ms}"] = int(len(kept_labels))
        if scalp_tensor.shape[1] < MIN_BIPOLAR_CHANNELS:
            skipped_cells[key] = f"only {scalp_tensor.shape[1]} scalp channels after mastoid exclusion"
        elif _degeneracy_reason(scalp_tensor) is not None:
            skipped_cells[key] = _degeneracy_reason(scalp_tensor)
        else:
            cells[key] = analyze_session("dandi_000574", "scalp_eeg", "scalp_eeg", patient, session_key,
                                          scalp_tensor.astype(float), bin_ms, is_point_process=False)

    record["cells"] = cells
    record["skipped_cells"] = skipped_cells
    return record


def build_all_sessions() -> list[dict]:
    root = data_root()
    sessions_out = []
    for reg_row in registry_sessions(root):
        session_key = reg_row["session"]
        with locked_json_update(CHECKPOINT_PATH) as checkpoint:
            entry = checkpoint.get(session_key)
            if entry is not None and entry.get("status") == "complete":
                sessions_out.append(entry["record"])
                continue
        print(f"fitting session: {session_key}", file=sys.stderr, flush=True)
        t0 = time.time()
        record = _process_session(reg_row)
        with locked_json_update(CHECKPOINT_PATH) as checkpoint:
            checkpoint[session_key] = {"status": "complete", "record": record}
        print(f"  {session_key}: stage={record.get('funnel_stage_reached')} in {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
        sessions_out.append(record)
    return sessions_out


# ---------------------------------------------------------------------------
# Funnel (3.1): n and patients/sessions lost at each restriction.
# ---------------------------------------------------------------------------

def funnel_summary(all_sessions: list[dict], single_unit_session_keys: set[tuple[str, str]]) -> dict:
    def _n_patients(records):
        return len(set(r["patient"] for r in records))

    all_recs = all_sessions
    depth_ok = [r for r in all_recs if r["funnel_stage_reached"] in ("depth_contact_survives", "depth_and_scalp_both_survive")]
    both_ok = [r for r in all_recs if r["funnel_stage_reached"] == "depth_and_scalp_both_survive"]
    both_ok_keys = {(r["patient"], r["session"]) for r in both_ok}
    single_unit_and_both = [r for r in both_ok if (r["patient"], r["session"]) in single_unit_session_keys]

    lost_at_depth = [{"patient": r["patient"], "session": r["session"], "reason": r.get("reason_stopped")}
                      for r in all_recs if r["funnel_stage_reached"] == "registered"]
    lost_at_scalp = [{"patient": r["patient"], "session": r["session"], "reason": r.get("reason_stopped")}
                      for r in all_recs if r["funnel_stage_reached"] == "depth_contact_survives"]

    return {
        "all_registry_sessions": {"n_sessions": len(all_recs), "n_patients": _n_patients(all_recs)},
        "depth_contact_survives": {
            "n_sessions": len(depth_ok), "n_patients": _n_patients(depth_ok),
            "n_sessions_lost_from_previous_stage": len(lost_at_depth),
            "n_patients_lost_entirely_at_this_stage": _n_patients(all_recs) - _n_patients(depth_ok),
            "lost": lost_at_depth,
        },
        "depth_and_scalp_both_survive": {
            "n_sessions": len(both_ok), "n_patients": _n_patients(both_ok),
            "n_sessions_lost_from_previous_stage": len(lost_at_scalp),
            "n_patients_lost_entirely_at_this_stage": _n_patients(depth_ok) - _n_patients(both_ok),
            "lost": lost_at_scalp,
        },
        "single_unit_also_survives_reference_only": {
            "n_sessions": len(single_unit_and_both), "n_patients": _n_patients(single_unit_and_both),
            "note": (
                "reference only, per the explicit instruction that the single-unit grain rides along and is "
                "never used to gate the depth-versus-scalp scope; sessions here are the intersection of "
                "depth_and_scalp_both_survive with the existing single-unit factor-analysis comparison's own "
                "session set"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Pre-declared branches (0.27, 0.33), fixed before either comparison's paired
# test is evaluated.
# ---------------------------------------------------------------------------

def predeclare_comparison_branches() -> dict:
    return {
        "deciding_quantity": (
            "the factor-analysis model's own observation-noise variance fraction "
            "(dimensionality.factor_analysis.observation_noise_variance_fraction, matched latent dimensionality "
            "across cells, scripts/run_latent_model_observation_noise_comparison.py's estimator), paired by "
            "session between the two cells of each comparison"
        ),
        "test": "two-sided paired sign-flip test (src/statistics.py paired_sign_flip_test), alpha=0.05, on (value of interest minus reference value) per paired session",
        "magnitude_threshold": (
            "no magnitude threshold is argued from the science for this comparison; the branch turns on the "
            "SIGN of the median paired difference and, only when that sign is unfavourable (positive -- the "
            "arm of interest is worse), on statistical significance at alpha=0.05. This is stated explicitly "
            "here, as its own field, rather than an invented comparator, because src/statistics.py's paired "
            "test has no magnitude threshold of its own to borrow."
        ),
        "decision_rule": (
            "let diff = median(value_of_interest - reference_value) across paired sessions. If diff <= 0 (the "
            "arm of interest sits at or below the reference -- the favourable or neutral direction), the "
            "branch is the 'costs little' branch regardless of significance. If diff > 0 and the paired "
            "sign-flip test is significant at p < 0.05, the branch is the 'is expensive'/'is the barrier' "
            "branch -- a significant result in the unfavourable direction is never folded into a null. If "
            "diff > 0 but the test is not significant, the branch is the 'no resolvable difference' branch "
            "and the minimum detectable paired difference (src/statistics.py minimum_detectable_paired_"
            "difference, 80% power) is reported beside it as a bound, never as a null result on its own."
        ),
        "comparison_a_band_effect_at_fixed_sensor": {
            "value_of_interest": "depth-contact LFP at the low band (8-45 Hz), same sensor, same contacts and the SAME trial set as the reference (Comparison A's own trial mask, matching the existing high-gamma cell's convention exactly)",
            "reference": "depth-contact LFP at the project's standard high-gamma band (70-150 Hz), the existing census/factor-analysis cell, read and never refit",
            "branches": {
                "band_restriction_costs_little": "the low-band cell's noise fraction sits at or below the high-gamma cell's",
                "band_restriction_is_expensive": "the low-band cell's noise fraction is materially higher",
                "no_resolvable_band_difference": "the difference sits inside the estimator's own floor",
            },
        },
        "comparison_b_sensor_effect_at_fixed_band": {
            "value_of_interest": "scalp EEG at the low band (8-45 Hz), same band, same patients and the SAME trial set as the reference (Comparison B's own trial mask, the intersection of the depth-contact and scalp grains' own recording validity)",
            "reference": "depth-contact LFP at the low band (8-45 Hz), refit with Comparison B's own trial mask so the two cells of this comparison share identical trials",
            "branches": {
                "sensor_costs_little_at_matched_band": "the scalp cell's noise fraction approaches the depth cell's once the band is matched",
                "sensor_is_the_barrier": "the scalp cell's noise fraction is materially higher than the depth cell's at the same band",
                "no_resolvable_sensor_difference": "the difference sits inside the estimator's own floor",
            },
        },
        "note": "pre-declared before either comparison's paired test was evaluated; resolved separately, never combined into one verdict",
    }


def _extract_noise_fraction(session_dict: dict | None) -> float | None:
    if session_dict is None:
        return None
    fa = session_dict.get("dimensionality", {}).get("factor_analysis", {})
    if fa.get("status") != "fitted":
        return None
    return fa.get("observation_noise_variance_fraction")


def _paired_quantity(
    sessions_a: dict[tuple[str, str], dict], sessions_b: dict[tuple[str, str], dict],
    extractor=_extract_noise_fraction,
) -> tuple[list[float], list[float], list[str]]:
    interest, reference, keys = [], [], []
    for key in sorted(set(sessions_a) & set(sessions_b)):
        va = extractor(sessions_a[key])
        vb = extractor(sessions_b[key])
        if va is None or vb is None:
            continue
        interest.append(va)
        reference.append(vb)
        keys.append(f"{key[0]}/{key[1]}")
    return interest, reference, keys


def resolve_comparison(
    sessions_a: dict, sessions_b: dict, rng: np.random.Generator,
    costs_little_name: str, expensive_name: str, no_resolvable_name: str,
    extractor=_extract_noise_fraction, extractor_description: str = "a fitted factor-analysis noise fraction",
) -> dict:
    interest, reference, keys = _paired_quantity(sessions_a, sessions_b, extractor)
    n_pairs = len(interest)
    if n_pairs < 4:
        return {
            "status": "not_computable", "n_pairs": n_pairs,
            "reason": f"fewer than 4 paired sessions ({n_pairs}) with {extractor_description} in both cells",
        }
    interest_arr, reference_arr = np.array(interest), np.array(reference)
    diffs = interest_arr - reference_arr
    test = paired_sign_flip_test(interest_arr, reference_arr, alternative="two-sided", rng=rng)
    median_diff = float(np.median(diffs))
    mdd = minimum_detectable_paired_difference(diffs)
    if median_diff <= 0:
        branch = costs_little_name
    elif test["p_value"] < 0.05:
        branch = expensive_name
    else:
        branch = no_resolvable_name
    return {
        "status": "fitted",
        "n_pairs": n_pairs,
        "paired_session_keys": keys,
        "r_obs_median_value_of_interest": float(np.median(interest_arr)),
        "r_obs_median_reference_value": float(np.median(reference_arr)),
        "median_difference_interest_minus_reference": median_diff,
        "mean_difference_interest_minus_reference": test["mean_diff"],
        "p_value": test["p_value"],
        "ci_lower_mean_difference": test["ci_lower"],
        "ci_upper_mean_difference": test["ci_upper"],
        "minimum_detectable_paired_difference_80pct_power": mdd,
        "branch": branch,
    }


# ---------------------------------------------------------------------------
# Descriptive per-cell summaries (participation ratio in both bases,
# persistence level in both bases) -- reported beside the deciding quantity,
# never as a tie-breaker.
# ---------------------------------------------------------------------------

def cell_descriptive_summary(sessions: list[dict]) -> dict:
    """Descriptive medians across a cell's own sessions. A PCA-basis
    participation ratio has no degeneracy guard in dimensionality_and_noise_
    term (unlike the factor-analysis arm, which reports its own status): a
    completely flat recording's covariance is all-zero, so its participation
    ratio is 0/0 -- not-a-number, serialised as null by this project's own
    JSON-safety convention (src/provenance.py's _json_safe). Every list here
    is therefore built with an explicit not-None filter rather than assuming
    a value is present just because its key is."""
    fa_fitted = [s for s in sessions if s.get("dimensionality", {}).get("factor_analysis", {}).get("status") == "fitted"]
    pca_pr = [v for s in sessions if (v := s.get("dimensionality", {}).get("pca", {}).get("participation_ratio")) is not None]
    fa_pr = [v for s in fa_fitted if (v := s["dimensionality"]["factor_analysis"].get("participation_ratio")) is not None]
    fa_noise = [v for s in fa_fitted if (v := s["dimensionality"]["factor_analysis"].get("observation_noise_variance_fraction")) is not None]
    pca_level = [v for s in sessions if s.get("persistence_contrast", {}).get("pca", {}).get("status") == "fitted"
                 and (v := s["persistence_contrast"]["pca"].get("level")) is not None]
    fa_level = [v for s in sessions if s.get("persistence_contrast", {}).get("factor_analysis", {}).get("status") == "fitted"
                and (v := s["persistence_contrast"]["factor_analysis"].get("level")) is not None]
    return {
        "n_sessions": len(sessions),
        "n_sessions_factor_analysis_fitted": len(fa_fitted),
        "n_sessions_pca_participation_ratio_not_a_number": len(sessions) - len(pca_pr),
        "median_participation_ratio_pca_basis": float(np.median(pca_pr)) if pca_pr else None,
        "median_participation_ratio_factor_analysis_basis": float(np.median(fa_pr)) if fa_pr else None,
        "median_factor_analysis_observation_noise_variance_fraction": float(np.median(fa_noise)) if fa_noise else None,
        "median_persistence_contrast_level_pca_basis": float(np.median(pca_level)) if pca_level else None,
        "n_sessions_persistence_level_fitted_pca_basis": len(pca_level),
        "median_persistence_contrast_level_factor_analysis_basis": float(np.median(fa_level)) if fa_level else None,
        "n_sessions_persistence_level_fitted_factor_analysis_basis": len(fa_level),
    }


# ---------------------------------------------------------------------------
# 3.7: where every attempted cell sits in the census's own margin ordering.
# ---------------------------------------------------------------------------

def census_margin_for_cell(admission_matrix: dict, dataset: str, modality: str, structure: str, bin_ms: int) -> dict:
    key = f"{dataset}::{modality}::{structure}::bin{bin_ms}"
    cell = admission_matrix.get(key)
    if cell is None:
        return {"status": "not_in_admission_matrix", "key": key}
    observables_of_interest = (
        "cross_validated_nugget_fraction", "participation_ratio_and_dimensionality_estimators",
        "persistence_contrast_d_perm_level_and_slope", "confinement_rate_lambda",
        "watts_strogatz_sigma_modularity_q_degree_assortativity",
    )
    return {
        "key": key, "n_sessions": cell["n_sessions"],
        "median_n_units_or_channels": cell["median_n_units_or_channels"],
        "median_n_trials": cell["median_n_trials"],
        "observable_margins": {name: cell["observables"].get(name) for name in observables_of_interest},
    }


def single_unit_reference_section(factor_model_artifact: dict, admission_matrix: dict) -> dict:
    """Read-only reference: the single-unit grain's already-computed cells
    (results/latent_model_observation_noise_comparison.json), pooled and per
    structure, never recomputed here and never folded into either
    comparison's branch."""
    per_structure = {}
    for key, cell in factor_model_artifact["cells"].items():
        if not key.startswith("dandi_000574::single_unit::"):
            continue
        structure = cell["structure"]
        bin_ms = cell["bin_ms"]
        per_structure.setdefault(structure, {})[f"bin{bin_ms}"] = {
            "summary": cell["summary"], "branch": cell["branch"],
            "observation_noise_term_comparison": cell["observation_noise_term_comparison"],
            "census_margin": census_margin_for_cell(admission_matrix, "dandi_000574", "single_unit", structure, bin_ms),
        }
    return {
        "feature_note": (
            "the single-unit grain's feature is spike counts (Anscombe-variance-stabilised, "
            "scripts/run_latent_model_comparison.py's convention), not a filtered voltage envelope -- it is "
            "therefore not band-matched to either the depth-contact or scalp low-band cells and is not folded "
            "into comparison A or comparison B's branch. It is reported here for reference only."
        ),
        "pooled_arm_flag": (
            "the 'pooled' structure mixes every recorded anatomical structure in a session into one "
            "population, which this project's own standing rule treats as a superseded baseline, never a "
            "substitute for a region-stratified fit -- reported here alongside the per-structure cells, not "
            "in place of them"
        ),
        "by_structure": per_structure,
    }


MOST_LIKELY_MISREADING = (
    "a difference between depth-contact LFP at the low band and scalp EEG at the low band is NOT evidence "
    "that scalp EEG cannot see this project's state in general. The two cells differ in sensor, distance from "
    "the source, volume conduction and channel count all at once, not in one isolated mechanism. What this "
    "decomposition buys is the separation of band from sensor as two SEPARATE questions with two separate "
    "answers; it does not isolate which single physical mechanism (distance, conduction, electrode count, or "
    "some combination) drives whatever gap comparison B finds."
)

NOT_RUN_BY_EXPLICIT_SCOPE = [
    "graph metrics (Watts-Strogatz sigma, modularity, degree, assortativity)",
    "gradient regression",
    "anatomical structure comparison (closed by an existing within-patient gate; not reopened here)",
    "Betti numbers / persistent homology",
    "dynamic mode decomposition (DMD)",
]


def main() -> None:
    root = data_root()
    census = json.loads(CENSUS_PATH.read_text())
    admission_matrix = census["admission_matrix"]
    factor_model_artifact = json.loads(FACTOR_MODEL_ARTIFACT_PATH.read_text())

    existing_high_gamma = load_existing_high_gamma_sessions()
    print(f"read {len(existing_high_gamma)} existing high-gamma depth-contact sessions (never refit)", file=sys.stderr)

    repro_check = reproducibility_check_against_existing_checkpoint(existing_high_gamma)
    print(f"reproducibility check against cached checkpoint: diff_is_empty={repro_check['diff_is_empty']}", file=sys.stderr)

    single_unit_session_keys = {
        (s["patient"], s["session"]) for s in
        json.loads(FACTOR_MODEL_CHECKPOINT_PATH.read_text())["dandi_000574"]["sessions"]
    }

    print("fitting the low-band depth and scalp cells, per session...", file=sys.stderr)
    all_sessions = build_all_sessions()
    funnel = funnel_summary(all_sessions, single_unit_session_keys)
    print(f"funnel: {funnel['all_registry_sessions']['n_sessions']} sessions -> "
          f"{funnel['depth_contact_survives']['n_sessions']} depth-survives -> "
          f"{funnel['depth_and_scalp_both_survive']['n_sessions']} depth-and-scalp-survive", file=sys.stderr)

    both_ok = [r for r in all_sessions if r["funnel_stage_reached"] == "depth_and_scalp_both_survive"]

    branch_rule = predeclare_comparison_branches()

    per_cell_sessions: dict[str, dict[tuple[str, str], dict]] = {
        "depth_low_band_comparison_a": {}, "depth_low_band_comparison_b": {}, "scalp_low_band": {},
    }
    for bin_ms in BIN_WIDTHS_MS:
        for r in both_ok:
            key = (r["patient"], r["session"])
            cells = r.get("cells", {})
            if f"depth_low_band_comparison_a_bin{bin_ms}" in cells:
                per_cell_sessions["depth_low_band_comparison_a"].setdefault(bin_ms, {})[key] = cells[f"depth_low_band_comparison_a_bin{bin_ms}"]
            if f"depth_low_band_comparison_b_bin{bin_ms}" in cells:
                per_cell_sessions["depth_low_band_comparison_b"].setdefault(bin_ms, {})[key] = cells[f"depth_low_band_comparison_b_bin{bin_ms}"]
            if f"scalp_low_band_bin{bin_ms}" in cells:
                per_cell_sessions["scalp_low_band"].setdefault(bin_ms, {})[key] = cells[f"scalp_low_band_bin{bin_ms}"]

    existing_high_gamma_by_bin: dict[int, dict[tuple[str, str], dict]] = {}
    for (patient, session, bin_ms), s in existing_high_gamma.items():
        existing_high_gamma_by_bin.setdefault(bin_ms, {})[(patient, session)] = s

    comparison_a_by_bin, comparison_b_by_bin = {}, {}
    cells_report: dict[str, dict] = {}
    for bin_ms in BIN_WIDTHS_MS:
        depth_a_sessions = per_cell_sessions["depth_low_band_comparison_a"].get(bin_ms, {})
        depth_b_sessions = per_cell_sessions["depth_low_band_comparison_b"].get(bin_ms, {})
        scalp_sessions = per_cell_sessions["scalp_low_band"].get(bin_ms, {})
        hg_sessions = existing_high_gamma_by_bin.get(bin_ms, {})

        comparison_a_by_bin[bin_ms] = resolve_comparison(
            depth_a_sessions, hg_sessions, _seed("comparison_a", bin_ms),
            "band_restriction_costs_little", "band_restriction_is_expensive", "no_resolvable_band_difference")
        comparison_b_by_bin[bin_ms] = resolve_comparison(
            scalp_sessions, depth_b_sessions, _seed("comparison_b", bin_ms),
            "sensor_costs_little_at_matched_band", "sensor_is_the_barrier", "no_resolvable_sensor_difference")

        cells_report[f"depth_contact_lfp_high_gamma_existing_bin{bin_ms}"] = {
            "role": "reference for comparison A, read from the existing artifact, never refit",
            "descriptive_summary": cell_descriptive_summary(list(hg_sessions.values())),
            "census_margin": census_margin_for_cell(admission_matrix, "dandi_000574", "depth_contact_lfp", "depth_contact_lfp", bin_ms),
        }
        cells_report[f"depth_contact_lfp_low_band_comparison_a_bin{bin_ms}"] = {
            "role": "value of interest for comparison A (band effect, fixed sensor); trial set matches the existing high-gamma cell exactly",
            "descriptive_summary": cell_descriptive_summary(list(depth_a_sessions.values())),
            "census_margin": census_margin_for_cell(admission_matrix, "dandi_000574", "depth_contact_lfp", "depth_contact_lfp", bin_ms),
        }
        cells_report[f"depth_contact_lfp_low_band_comparison_b_bin{bin_ms}"] = {
            "role": "reference for comparison B (sensor effect, fixed band); trial set matched to the scalp grain",
            "descriptive_summary": cell_descriptive_summary(list(depth_b_sessions.values())),
            "census_margin": census_margin_for_cell(admission_matrix, "dandi_000574", "depth_contact_lfp", "depth_contact_lfp", bin_ms),
        }
        cells_report[f"scalp_eeg_low_band_bin{bin_ms}"] = {
            "role": "value of interest for comparison B (sensor effect, fixed band); the only cell this scalp grain can represent at all",
            "descriptive_summary": cell_descriptive_summary(list(scalp_sessions.values())),
            "census_margin": census_margin_for_cell(admission_matrix, "dandi_000574", "scalp_eeg", "scalp_eeg", bin_ms),
        }

    payload = {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "trigger": (
            "does the non-invasive tier's death trace to frequency content (high-gamma power is not "
            "representable in a non-invasive recording's band), to sensor physics (electrode count, distance, "
            "volume conduction), or both -- decomposed within one corpus, one set of patients, one set of "
            "trials, rather than inferred across different corpora with different instruments"
        ),
        "precursor": {
            "artifacts": [
                "results/observability_and_power_census.json (admission matrix and margins, including the "
                "scalp-EEG cells this script reads but does not refit)",
                "results/latent_model_observation_noise_comparison.json (the existing depth-contact high-gamma "
                "factor-analysis cell and the single-unit reference cells, read and never refit)",
            ],
            "settles": (
                "which (dataset, modality, structure, bin width) cells are measurable at all and by what "
                "margin; the depth-contact-lfp cell's factor-analysis noise fraction at the high-gamma band; "
                "the single-unit grain's per-structure factor-analysis cells"
            ),
            "gap_this_stage_fills": (
                "the low band (8-45 Hz) at both sensors, and the two paired comparisons this enables -- "
                "neither existed before this script"
            ),
        },
        "reproducibility_check": repro_check,
        "low_band_hz": [LOW_BAND_LO_HZ, LOW_BAND_HI_HZ],
        "low_band_provenance": (
            "fixed before anything was fit, by two physical constraints, not tuned: entirely below the 50 Hz "
            "mains notch already applied in this corpus (harmonics at 100 and 150 Hz), and entirely below the "
            "scalp series' own Nyquist frequency (~69.9 Hz) with margin"
        ),
        "funnel": funnel,
        "predeclared_branches": branch_rule,
        "cells": cells_report,
        "comparison_a_band_effect_at_fixed_sensor": {"bin100": comparison_a_by_bin[100], "bin200": comparison_a_by_bin[200]},
        "comparison_b_sensor_effect_at_fixed_band": {"bin100": comparison_b_by_bin[100], "bin200": comparison_b_by_bin[200]},
        "single_unit_reference": single_unit_reference_section(factor_model_artifact, admission_matrix),
        "most_likely_misreading": MOST_LIKELY_MISREADING,
        "observables_not_run_by_explicit_scope": NOT_RUN_BY_EXPLICIT_SCOPE,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"wrote {OUTPUT_PATH}", file=sys.stderr)
    print(json.dumps({
        "comparison_a_bin100_branch": comparison_a_by_bin[100].get("branch", comparison_a_by_bin[100].get("status")),
        "comparison_a_bin200_branch": comparison_a_by_bin[200].get("branch", comparison_a_by_bin[200].get("status")),
        "comparison_b_bin100_branch": comparison_b_by_bin[100].get("branch", comparison_b_by_bin[100].get("status")),
        "comparison_b_bin200_branch": comparison_b_by_bin[200].get("branch", comparison_b_by_bin[200].get("status")),
    }, indent=2))


if __name__ == "__main__":
    main()
