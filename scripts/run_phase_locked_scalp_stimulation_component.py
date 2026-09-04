#!/usr/bin/env python3
"""Does the phase of an ongoing scalp oscillation at the moment of closed-loop
stimulation delivery change the rate-free state-deviation component this
project has identified as predicting behaviour -- and, if so, at which phase?

This is the only human corpus in the project carrying oscillatory-phase
timing: closed-loop, phase-tuned alternating-current stimulation to the
scalp, locked in real time to each participant's own endogenous alpha
oscillation, at six evenly spaced phase lags, with an active group
stimulated over the recorded source and a control group stimulated away
from it. Every other timing result this project holds comes from corpora
that deliver stimulation in task time (which item, which epoch); none of
them can be asked at within-cycle resolution.

Four questions, in order, each gating the next:

  presence -- is the component present in this corpus at all (stimulation-OFF
             baseline, rotation null), compared against the scalp-tier
             reference the recording-tier analysis measured in a different human
             corpus.
  behaviour link -- does the component predict accuracy in this maintenance
             delay (stimulation-OFF baseline, all trials with an outcome),
             compared against the three-corpus invasive human null.
  phase modulation -- does the stimulation phase modulate the component
             (frozen baseline coordinate frame, stimulation-ON trials), with
             three mandatory artifact controls beside every number: the
             control group, a band away from the stimulation band, and the
             same test on the stimulation-OFF block where the trigger codes
             exist but no current is delivered.
  benefit prediction -- does the baseline-block component predict who
             benefits from stimulation and how far the component moves
             between blocks.

Reused machinery, unmodified (imported, never edited -- see each function's
own module for the argument this analysis is not re-litigating):
  run_haslacher_phase_omega        -- participant lists, data directory,
                                       phase-condition code map, trial-outcome
                                       reader, accuracy-by-phase harmonic fit.
  run_haslacher_stimulation_geometry -- participant-native (author-ordered)
                                       preprocessing and retention-window
                                       trial extraction.
  run_haslacher_phase_diffusion    -- circular-harmonic fit and the
                                       population-level circular vector test.
  run_recording_tier_component_transfer -- the component estimator
                                       (rate_free_state_deviation), its
                                       magnitude-matched rotation null, and
                                       the reference effect the presence and
                                       behaviour-link tests use.

Two pieces of machinery this analysis needs but the reused modules do not provide
are written here, each a close variant of a reused function rather than a
fresh design: a channel-rejection-and-SASS pipeline generalised to an
arbitrary passband (the reused participant-native preprocessing hard-codes
the 8-14 Hz analysis band, needed for the artifact-band control), and a
trial-outcome reader pointed at the stimulation-OFF block (the reused reader
is hard-coded to the stimulation-ON block, needed because the behaviour-link
test asks about the OFF block specifically).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_phase_locked_scalp_stimulation_component.py
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
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import mne  # noqa: E402
from pyprep.find_noisy_channels import NoisyChannels  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from corpus_sessions import data_root  # noqa: E402
from preprocessing import band_power  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci, minimum_detectable_paired_difference, partial_correlation_permutation_test,
    permutation_pvalue, stable_seed,
)
from run_haslacher_phase_omega import (  # noqa: E402
    ACTIVE_SUBJECTS, CONTROL_SUBJECTS, DATA_DIR, PHASE_CONDITIONS, RETENTION_TMAX, RETENTION_TMIN,
    modulation_from_outcomes, trial_outcomes,
)
from run_haslacher_stimulation_geometry import (  # noqa: E402
    AUX_CHANNELS, NOT_OF_INTEREST, PROTECT, SATURATION_THRESHOLD, SFREQ_ANALYSIS,
    _preprocess_author_native, _retention_trials, _sass,
)
from run_haslacher_phase_diffusion import (  # noqa: E402
    active_control_difference, bin_analog_trials, group_vector_test, harmonic_coefficients,
)
from run_recording_tier_component_transfer import (  # noqa: E402
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS, N_ROTATION_NULL_DRAWS, _bias_only_values,
    _patient_clustered_test, block_a_tier as presence_test_tier, discover_000574_sessions, load_000574_session_tiers,
    rate_free_state_deviation, rotation_null_variance_test,
)

OUTPUT_PATH = ROOT / "results" / "phase_locked_scalp_stimulation_component.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_phase_locked_scalp_stimulation_component"

STIMULATION_BAND_HZ = (8.0, 14.0)  # this dataset's own analysis band (Data/README.md sec 4)
ARTIFACT_CONTROL_BAND_HZ = (35.0, 45.0)  # away from the stimulation band and from every 50 Hz
                                          # mains harmonic, so this control needs no notch step
N_COMPONENTS_FRAME = 3  # matches the delivered diffusion script's own frozen-frame dimensionality
N_PERM_PHASE_MODULATION = 5000  # the required minimum for the within-participant phase-label-shuffle null
MIN_BASELINE_TRIALS = 10
MIN_TRIALS_WITH_OUTCOME = 10
MIN_TRIALS_PER_PHASE_CONDITION = 5  # the delivered diffusion script's own floor
MIN_PARTICIPANTS_FOR_TEST = 5  # the sign-flip test's own attainable-p floor (see _patient_clustered_test)

PRESENCE_DECISION_RULE = (
    "Per participant, on the stimulation-OFF baseline block only: build one feature per trial (mean "
    "8-14 Hz Hilbert-envelope power per scalp channel over the whole retention-window epoch, the same "
    "per-channel band-power construction the recording-tier analysis's own scalp_eeg tier used), then test the "
    "rate-free state-deviation component's variance across trials against a magnitude-matched rotation "
    "null (rotation_null_variance_test, unchanged). Effects are pooled across participants with the "
    "two-sided paired sign-flip test (the same primitive the recording-tier analysis uses). If fewer than 5 "
    "participants contribute a computed effect, the sign-flip test cannot structurally reach p<=0.05 and "
    "the branch is 'inconclusive_below_detection_floor'. Otherwise: significant (p<=0.05) -> "
    "'the_component_is_present_in_the_non_invasive_timing_corpus'. Not significant: the minimum detectable "
    "paired difference (80% power) is compared against the recording-tier analysis's own scalp-tier reference "
    "effect (results/recording_tier_component_transfer.json, the scalp_eeg tier's own presence-test entry, read "
    "once before this analysis's own number is computed). MDD below that reference -> "
    "'the_component_is_absent_in_the_non_invasive_timing_corpus'. MDD at or above it -> "
    "'inconclusive_below_detection_floor'. If the absent branch fires, the phase-modulation and "
    "benefit-prediction group-level synthesis do not run; the behaviour-link test still runs, since it "
    "does not depend on this gate."
)

BEHAVIOUR_LINK_DECISION_RULE = (
    "Per participant, on the stimulation-OFF baseline block, on every trial carrying a recovered outcome "
    "(correct/incorrect, matched to its own epoch by trigger onset sample rather than assumed to line up "
    "positionally): the point-biserial correlation of accuracy against the component's per-trial value "
    "(partial_correlation_permutation_test, zero controls), pooled across participants with the two-sided "
    "paired sign-flip test. Mandatory control: the identical statistic recomputed with every trial's "
    "component value replaced by that participant's own leave-one-out mean over every OTHER trial "
    "(_bias_only_values, unchanged). If the control reproduces significance with the same sign as the "
    "primary result, the branch is void: "
    "'component_behaviour_link_not_separable_from_a_participant_level_offset'. Otherwise: fewer than 5 "
    "contributing participants -> 'inconclusive_below_detection_floor'. Primary significant -> "
    "'the_component_predicts_accuracy_in_a_non_invasive_human_maintenance_delay'. Primary not significant: "
    "MDD compared against the fixed reference of 0.14 r units (the project's standing behavioural "
    "reference). MDD below 0.14 -> "
    "'the_component_does_not_predict_accuracy_in_a_non_invasive_human_maintenance_delay'. MDD at or above "
    "0.14 -> 'inconclusive_below_detection_floor'. Whichever fires is reported against the delivered "
    "three-corpus invasive human null and against the non-human effect measured at matched human error "
    "counts (both read from results/human_maintenance_behaviour_link.json)."
)

PHASE_MODULATION_DECISION_RULE = (
    "Per participant: freeze a channel-z-scored 3-component PCA coordinate frame on the participant's own "
    "stimulation-OFF baseline (bin_analog_trials + PCA.fit, unchanged from the delivered diffusion "
    "script's own construction, fit without any behavioural outcome), project stimulation-ON "
    "retention-window trials into that frame, average each trial's projected trajectory over the retention "
    "window to one latent vector per trial, and compute the rate-free state-deviation component once over "
    "the whole pooled set of a participant's stimulation-ON trials (not separately per phase condition, so "
    "the leave-one-out reference direction is not itself conditioned on phase). Per-phase-condition means "
    "of that per-trial component are fit with one circular harmonic (harmonic_coefficients, unchanged); "
    "modulation depth is the fitted amplitude. Significance is a permutation null that reshuffles the "
    "trial-to-phase-condition labels within participant, 5000 draws, seeded by a stable hash of a "
    "descriptive tag. Per-participant harmonic vectors are pooled within group by the delivered "
    "participant-level circular-rotation population test (group_vector_test, unchanged); the between-group "
    "test is the delivered vector-difference permutation test (active_control_difference, unchanged), with "
    "a minimum detectable amplitude difference from the two-sample normal approximation on the "
    "participant-level amplitude scalars. Three artifact controls run identically for every group: the "
    "control group at the same six lags with the same hardware; the identical pipeline and test on a band "
    "away from the stimulation band (35-45 Hz) and clear of every 50 Hz mains harmonic; and the identical "
    "pipeline and test on the stimulation-OFF block's own six-phase-condition trigger codes, where the "
    "codes exist but no current was ever delivered. Branches: both groups' population tests significant -> "
    "'phase_modulation_is_present_in_both_groups' (an artifact verdict, not a biological one). Active "
    "group's population test significant, between-group test significant, and neither the control-band nor "
    "the baseline-block control reproduces significance in the active group -> "
    "'the_stimulation_phase_modulates_the_component_in_the_targeted_group_only'. Active group's population "
    "test not significant and the minimum detectable amplitude (one-sample, against zero, on the active "
    "group's participant-level amplitude scalars) is below this corpus's own behavioural accuracy-by-phase "
    "modulation depth (the delivered accuracy-by-phase harmonic, population-pooled in the active group) -> "
    "'the_stimulation_phase_does_not_modulate_the_component'. Anything else -> "
    "'inconclusive_below_detection_floor'."
)

BENEFIT_PREDICTION_DECISION_RULE = (
    "Runs only if the presence test's group-level branch is "
    "'the_component_is_present_in_the_non_invasive_timing_corpus'. Per participant, from the "
    "stimulation-OFF baseline block alone: the component's central value (median) and spread (IQR) from "
    "the presence test's own per-trial deviation array, and the presence test's own presence statistic "
    "(the participant's "
    "signed observed-minus-null variance). Between-participant Pearson correlation "
    "(partial_correlation_permutation_test, zero controls) of each of these three predictors against two "
    "outcomes: this corpus's own accuracy-by-phase behavioural modulation depth (delivered, "
    "modulation_from_outcomes on the stimulation-ON block), and the displacement of the component between "
    "blocks (mean stimulation-ON per-trial deviation, in the frozen frame, minus mean stimulation-OFF "
    "per-trial deviation in the same frame). Active group is primary, control group reported beside it. "
    "For every cell: significant (p<=0.05) -> 'the_predictor_predicts_the_outcome_across_participants'. Not "
    "significant and the minimum detectable correlation (Fisher-z normal approximation at this n) is below "
    "0.14 -> 'no_link_above_the_reported_bound'. Otherwise -> 'underpowered_to_ask', the expected outcome at "
    "n=21, and never reported as a plain null."
)


def minimum_detectable_correlation(n: int, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Smallest true Pearson r an independent-samples correlation at this n could detect at the given
    power, via the Fisher z transform's normal approximation -- the correlation analogue of
    statistics.minimum_detectable_paired_difference, which is built for paired mean differences and does
    not apply to a between-participant correlation coefficient."""
    from scipy.stats import norm
    if n < 4:
        return {"status": "not_computable", "n": n}
    z_factor = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    z_effect = z_factor / np.sqrt(n - 3)
    return {"status": "computed", "n": n, "alpha": alpha, "power": power,
            "z_factor": z_factor, "mdd_r": float(np.tanh(z_effect))}


def minimum_detectable_group_difference(n1: int, n2: int, pooled_sd: float,
                                          alpha: float = 0.05, power: float = 0.80) -> dict:
    """Smallest true two-sample mean difference this design could detect at the given power, from the
    observed pooled standard deviation -- the two-sample analogue of
    statistics.minimum_detectable_paired_difference, needed because the phase-modulation test's
    between-group amplitude contrast is not a paired design."""
    from scipy.stats import norm
    if n1 < 2 or n2 < 2 or not np.isfinite(pooled_sd):
        return {"status": "not_computable", "n1": n1, "n2": n2}
    z_factor = float(norm.ppf(1 - alpha / 2) + norm.ppf(power))
    mdd = z_factor * pooled_sd * np.sqrt(1.0 / n1 + 1.0 / n2)
    return {"status": "computed", "n1": n1, "n2": n2, "pooled_sd": float(pooled_sd),
            "alpha": alpha, "power": power, "z_factor": z_factor, "mdd": float(mdd)}


def pooled_sd(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    return float(np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)))


def circular_distance_deg(a: float, b: float) -> float:
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ── Reference context, read-only, from sibling artifacts (never edited, never recomputed here) ──────────

def recording_tier_scalp_reference() -> dict:
    doc = json.loads((ROOT / "results" / "recording_tier_component_transfer.json").read_text())
    pooled_test = doc["block_a"]["scalp_eeg"]["pooled_patient_test"]
    return {"mean_value": pooled_test["mean_value"], "abs_mean_value": abs(pooled_test["mean_value"]),
            "mdd": pooled_test.get("mdd"), "n_patients": pooled_test["n_patients"],
            "p_value": pooled_test["p_value"],
            "source": "results/recording_tier_component_transfer.json (scalp_eeg tier, pooled_patient_test)"}


def three_corpus_human_null_context() -> dict:
    doc = json.loads((ROOT / "results" / "human_maintenance_behaviour_link.json").read_text())
    combined = doc["block_b"]["combined_within_load_then_meta_analysed"]
    branch = doc["block_b"]["branch"]
    secondary = doc["block_b"]["secondary_pooled_across_load_all_corpora"]
    non_human = doc["block_c"]["human_distribution_matched_draws"]["pooled"]
    return {
        "combined_raw_r": combined["raw"]["pooled"], "combined_raw_p": combined["raw"]["p_value"],
        "combined_joint_partial_r": combined["joint_partial"]["pooled"],
        "combined_joint_partial_p": combined["joint_partial"]["p_value"],
        "minimum_detectable_difference": branch["minimum_detectable_difference_80pct_power"]["mdd"],
        "n_patients": secondary["n_patients"], "delivered_branch": branch["branch"],
        "non_human_effect_at_matched_human_error_counts_r": non_human["mean_value"],
        "non_human_effect_at_matched_human_error_counts_p": non_human["p_value"],
        "source": "results/human_maintenance_behaviour_link.json",
    }


# ── Reproduction gate ──────────────────────────────────────────────────────────────────────────────────

def reproduction_gate() -> dict:
    """Re-runs the delivered component estimator on the delivered recording-tier scalp sessions (from
    that analysis's own checkpoints; no new preprocessing) and checks the pooled scalp-tier test matches the
    delivered artifact at 1e-6."""
    root = data_root()
    sessions = discover_000574_sessions(root)
    per_tier = []
    for patient, session_key, nwb_path in sessions:
        tiers = load_000574_session_tiers(patient, session_key, nwb_path)
        rec = tiers.get("scalp_eeg")
        if rec is None or rec.get("status") != "computed":
            continue
        per_tier.append({"patient": patient, "session_key": session_key,
                          "deviation": rec["deviation"], "gate": rec["gate"]})
    reproduced = presence_test_tier(per_tier, None)["pooled_patient_test"]
    delivered = json.loads((ROOT / "results" / "recording_tier_component_transfer.json").read_text()
                            )["block_a"]["scalp_eeg"]["pooled_patient_test"]
    fields = ["mean_value", "ci_lower", "ci_upper", "mdd", "p_value"]
    diffs = {f: abs(reproduced[f] - delivered[f]) for f in fields}
    matched = max(diffs.values()) < 1e-6 and reproduced["n_patients"] == delivered["n_patients"]
    return {"status": "matched" if matched else "mismatch", "tolerance": 1e-6, "max_abs_diff": max(diffs.values()),
            "per_field_abs_diff": diffs, "reproduced": reproduced, "delivered": delivered}


# ── Preprocessing generalised to an arbitrary passband (control-band artifact test) ──────────────────────

def _preprocess_fixed_band(subject: str, group: str, lo: float, hi: float) -> tuple:
    """Same channel-rejection, SASS and average-reference pipeline as
    run_haslacher_stimulation_geometry._preprocess_author_native, generalised to an arbitrary passband --
    needed because that function hard-codes the 8-14 Hz analysis band and must not be modified. Every
    channel-rejection and SASS decision is identical to the primary band's pipeline; only the final filter
    edges differ, so a phase modulation that survives here but not in the primary band is diagnostic of
    the stimulation band specifically rather than of the pipeline in general. The alpha-specific SASS
    sanity check (_target_alpha_power, fixed internally to 8-14 Hz) does not generalise to another band and
    is not reused here; SASS's own component count is carried instead, as a weaker but band-agnostic
    diagnostic."""
    no_stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "no_stim.vhdr"), preload=True, verbose="ERROR")
    stim = mne.io.read_raw_brainvision(str(DATA_DIR / subject / "stim.vhdr"), preload=True, verbose="ERROR")
    drop_hint = NOT_OF_INTEREST[group]

    probe = no_stim.copy().drop_channels([c for c in AUX_CHANNELS + drop_hint if c in no_stim.ch_names])
    noisy = NoisyChannels(probe, random_state=stable_seed(f"phase_locked_pyprep_{subject}_{lo}_{hi}"))
    noisy.find_all_bads(ransac=False)
    bad_noisy = noisy.get_bads()

    stim_probe = stim.copy().drop_channels([c for c in AUX_CHANNELS + drop_hint if c in stim.ch_names])
    maxv = np.abs(stim_probe.get_data()).max(axis=-1)
    bad_saturated = [c for c, v in zip(stim_probe.ch_names, maxv) if v > SATURATION_THRESHOLD]
    requested = (set(bad_noisy) | set(bad_saturated) | set(drop_hint) | {"stim"}) - set(PROTECT)
    dropped = sorted(c for c in requested if c in no_stim.ch_names and c in stim.ch_names)
    no_stim.drop_channels(dropped)
    stim.drop_channels(dropped)

    no_stim.resample(SFREQ_ANALYSIS)
    stim.resample(SFREQ_ANALYSIS)
    no_stim.filter(lo, hi, verbose="ERROR")
    stim.filter(lo, hi, verbose="ERROR")
    n_sass = _sass(no_stim, stim)

    for raw in (no_stim, stim):
        raw.drop_channels([c for c in AUX_CHANNELS if c in raw.ch_names])
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    return no_stim, stim, {"channels_dropped": dropped, "n_sass_components_removed": n_sass}


# ── Outcome reader for the stimulation-OFF block ──────────────────────────────────────────────────────────

def _outcomes_by_onset_sample(raw: "mne.io.Raw") -> dict[int, int]:
    """Maps each phase-condition trigger's onset sample (in the given raw object's own, possibly
    resampled, sample grid) to its correctness -- identical trial-outcome logic to
    run_haslacher_phase_omega.trial_outcomes, generalised to accept an already-loaded Raw object instead
    of always re-reading the stimulation-ON file from disk, and keyed by onset sample (not list position)
    so it can be matched exactly to the trials an mne.Epochs call admits from the SAME raw object, which
    may drop boundary trials trial_outcomes' own event scan would not."""
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    sfreq = raw.info["sfreq"]
    out = {}
    for i in range(len(events) - 1):
        code = events[i, 2]
        if code not in PHASE_CONDITIONS:
            continue
        rt = (events[i + 1, 0] - events[i, 0]) / sfreq - 3.8
        if rt < 0.75:
            out[int(events[i, 0])] = int(events[i + 1, 2] == 10)
    return out


def _epochs_with_onsets(raw: "mne.io.Raw", codes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Same mne.Epochs construction as run_haslacher_stimulation_geometry._retention_trials, but also
    returns each admitted trial's onset sample so outcomes can be matched exactly (see
    _outcomes_by_onset_sample)."""
    events, _ = mne.events_from_annotations(raw, verbose="ERROR")
    epochs = mne.Epochs(raw, events, event_id=codes, tmin=RETENTION_TMIN, tmax=RETENTION_TMAX,
                        baseline=None, preload=True, on_missing="ignore", verbose="ERROR")
    return epochs.get_data(copy=True), epochs.events[:, 0]


# ── Feature construction ──────────────────────────────────────────────────────────────────────────────────

def _channel_band_power_features(epochs: np.ndarray, srate: float, lo: float, hi: float) -> np.ndarray:
    """Per-trial mean band power per channel, over the whole epoch (already windowed to the retention
    window by _retention_trials) -- the same per-channel Hilbert-envelope band-power construction the
    recording-tier analysis's own scalp_eeg tier used, applied here to whichever band this analysis calls it with."""
    n_trials = epochs.shape[0]
    out = np.full((n_trials, epochs.shape[1]), np.nan)
    for i in range(n_trials):
        sig_tc = epochs[i].T  # (T, C)
        out[i] = band_power(sig_tc, (lo, hi), srate).mean(axis=0)
    return out


def _fit_frozen_frame(baseline_trials: np.ndarray, srate: float) -> dict:
    """Fits center/scale/PCA on the pooled, binned stimulation-OFF baseline -- the identical construction
    run_haslacher_phase_diffusion.analyze_subject uses to build its own frozen coordinate frame, reused
    here rather than re-derived."""
    binned = bin_analog_trials(baseline_trials, srate)
    observations = binned.transpose(0, 2, 1).reshape(-1, binned.shape[1])
    center = observations.mean(axis=0)
    scale = observations.std(axis=0)
    scale[scale < 1e-10] = 1.0
    pca = PCA(n_components=min(N_COMPONENTS_FRAME, observations.shape[1]))
    pca.fit((observations - center) / scale)
    return {"center": center, "scale": scale, "pca": pca}


def _project_mean_latent(trials: np.ndarray, frame: dict, srate: float) -> np.ndarray:
    """Projects trials into the frozen frame and averages each trial's latent trajectory over the
    retention window to one vector per trial -- mirrors
    run_haslacher_phase_diffusion._phase_diffusion's own binning and projection step, stopping short of
    that function's residual/diffusion fit because this analysis wants the raw per-trial latent mean."""
    binned = bin_analog_trials(trials, srate)
    flat = ((binned.transpose(0, 2, 1) - frame["center"]) / frame["scale"]).reshape(-1, len(frame["center"]))
    latent = frame["pca"].transform(flat).reshape(len(binned), binned.shape[-1], -1)
    return latent.mean(axis=1)


# ── Phase modulation, one condition (primary band / control band / baseline-block control) ──────────

def _harmonic_permutation_test(deviation: np.ndarray, pooled_codes: np.ndarray, seed_tag: str,
                                n_permutations: int = N_PERM_PHASE_MODULATION) -> dict | None:
    """Fits one circular harmonic to the six phase-condition means of a per-trial deviation array, and
    tests its amplitude against a within-participant, phase-label-shuffle permutation null -- the core
    statistical ladder shared by every phase-modulation condition (primary band, control band,
    stimulation-OFF control), factored out so it can be exercised directly on synthetic per-trial values
    without needing a raw EEG epoch array or a fitted coordinate frame. Returns None if any phase
    condition has a non-finite mean (too few or all-NaN trials in that condition)."""
    codes_ordered = sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c])
    means_obs = {c: float(np.nanmean(deviation[pooled_codes == c])) for c in codes_ordered}
    if any(not np.isfinite(v) for v in means_obs.values()):
        return None
    harmonic_obs = harmonic_coefficients(means_obs)
    rng = np.random.default_rng(stable_seed(seed_tag))
    null_amp = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled = rng.permutation(pooled_codes)
        means_p = {c: (float(np.nanmean(deviation[shuffled == c])) if np.any(shuffled == c) else np.nan)
                   for c in codes_ordered}
        h_p = harmonic_coefficients(means_p) if all(np.isfinite(v) for v in means_p.values()) else None
        null_amp[i] = h_p["amplitude"] if h_p is not None else np.nan
    valid = null_amp[np.isfinite(null_amp)]
    p_value = permutation_pvalue(valid >= harmonic_obs["amplitude"]) if len(valid) else float("nan")
    return {"harmonic": harmonic_obs, "p_value": p_value, "n_permutations_valid": int(len(valid)),
            "condition_means": {str(c): v for c, v in means_obs.items()}}


def _phase_modulation_condition(frame: dict, trials_by_phase: dict[int, np.ndarray], srate: float,
                        seed_tag: str) -> dict:
    codes_ordered = sorted(PHASE_CONDITIONS, key=lambda c: PHASE_CONDITIONS[c])
    phase_counts = {str(c): int(len(trials_by_phase.get(c, []))) for c in codes_ordered}
    if any(phase_counts[str(c)] < MIN_TRIALS_PER_PHASE_CONDITION for c in codes_ordered):
        return {"status": "excluded", "reason": "at_least_one_phase_condition_has_fewer_than_five_trials",
                "phase_trial_counts": phase_counts}
    pooled_trials = np.concatenate([trials_by_phase[c] for c in codes_ordered], axis=0)
    pooled_codes = np.concatenate([np.full(len(trials_by_phase[c]), c) for c in codes_ordered])
    latent_mean = _project_mean_latent(pooled_trials, frame, srate)
    deviation = rate_free_state_deviation(latent_mean)
    fitted = _harmonic_permutation_test(deviation, pooled_codes, seed_tag)
    if fitted is None:
        return {"status": "not_computable", "reason": "non_finite_condition_mean",
                "phase_trial_counts": phase_counts}
    return {"status": "computed", "harmonic": fitted["harmonic"], "p_value": fitted["p_value"],
            "n_permutations_valid": fitted["n_permutations_valid"], "phase_trial_counts": phase_counts,
            "condition_means": fitted["condition_means"],
            "n_trials_total": int(len(pooled_codes)),
            "mean_deviation": float(np.nanmean(deviation))}


# ── Per-participant pipeline (checkpointed unit) ─────────────────────────────────────────────────────────

def process_participant(subject: str, group: str) -> dict:
    try:
        no_stim, stim, qc = _preprocess_author_native(subject, group)
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        return {"status": "refused", "subject": subject, "group": group,
                "reason": f"preprocessing_failed:{type(error).__name__}:{error}"}

    baseline_pooled = _retention_trials(no_stim)[0]
    if baseline_pooled.shape[0] < MIN_BASELINE_TRIALS:
        return {"status": "refused", "subject": subject, "group": group,
                "reason": "too_few_baseline_trials", "n_baseline_trials": int(baseline_pooled.shape[0])}

    srate = float(no_stim.info["sfreq"])
    baseline_by_phase = _retention_trials(no_stim, codes=list(PHASE_CONDITIONS))
    stim_by_phase = _retention_trials(stim, codes=list(PHASE_CONDITIONS))

    # ---- presence test: alpha band, per-channel features, baseline ----
    activity_a = _channel_band_power_features(baseline_pooled, srate, *STIMULATION_BAND_HZ)
    deviation_a = rate_free_state_deviation(activity_a)
    rng_a = np.random.default_rng(stable_seed(f"phase_locked_presence|{subject}"))
    gate_a = rotation_null_variance_test(activity_a, N_ROTATION_NULL_DRAWS, rng_a)
    finite_a = deviation_a[np.isfinite(deviation_a)]
    presence_record = {
        "gate": gate_a, "n_trials": int(activity_a.shape[0]),
        "median_deviation": float(np.median(finite_a)) if finite_a.size else None,
        "iqr_deviation": ([float(np.percentile(finite_a, 25)), float(np.percentile(finite_a, 75))]
                           if finite_a.size else None),
    }

    # ---- behaviour-link test: alpha band, per-channel features + matched outcomes, baseline ----
    epochs_b, onsets_b = _epochs_with_onsets(no_stim, list(PHASE_CONDITIONS))
    outcome_map = _outcomes_by_onset_sample(no_stim)
    correct_b = np.array([outcome_map.get(int(s), np.nan) for s in onsets_b])
    has_outcome = np.isfinite(correct_b)
    n_errors = int(np.sum(correct_b[has_outcome] == 0)) if has_outcome.any() else 0
    n_with_outcome = int(has_outcome.sum())
    if n_with_outcome >= MIN_TRIALS_WITH_OUTCOME and epochs_b.shape[0] == len(correct_b):
        activity_b = _channel_band_power_features(epochs_b, srate, *STIMULATION_BAND_HZ)
        deviation_b = rate_free_state_deviation(activity_b)
        usable = np.isfinite(deviation_b) & has_outcome
        if int(usable.sum()) >= MIN_TRIALS_WITH_OUTCOME:
            rng_p = np.random.default_rng(stable_seed(f"phase_locked_behaviour_link_primary|{subject}"))
            primary = partial_correlation_permutation_test(correct_b[usable], deviation_b[usable],
                                                             controls=[], n_perm=2000, rng=rng_p)
            bias_values = _bias_only_values(deviation_b)
            bias_usable = np.isfinite(bias_values) & has_outcome
            if int(bias_usable.sum()) >= MIN_TRIALS_WITH_OUTCOME:
                rng_b = np.random.default_rng(stable_seed(f"phase_locked_behaviour_link_bias|{subject}"))
                bias = partial_correlation_permutation_test(correct_b[bias_usable], bias_values[bias_usable],
                                                              controls=[], n_perm=2000, rng=rng_b)
            else:
                bias = {"status": "not_computable", "reason": "too_few_trials_with_a_defined_loo_mean"}
            behaviour_link_record = {"status": "computed", "n_trials": int(usable.sum()), "n_errors": n_errors,
                               "primary": primary, "bias_only": bias}
        else:
            behaviour_link_record = {"status": "too_few_trials", "n_trials": int(usable.sum()), "n_errors": n_errors}
    else:
        behaviour_link_record = {"status": "too_few_trials_with_outcome", "n_trials_with_outcome": n_with_outcome,
                           "n_errors": n_errors}

    # ---- phase-modulation test, primary: frozen alpha frame, stimulation-ON trials ----
    frame_alpha = _fit_frozen_frame(baseline_pooled, srate)
    phase_modulation_primary_result = _phase_modulation_condition(frame_alpha, stim_by_phase, srate, f"phase_locked_phase_modulation_primary|{subject}")

    # ---- phase-modulation test, baseline-block control: same frame, stimulation-OFF trials (trigger codes exist, no current) ----
    phase_modulation_stimulation_off_control_result = _phase_modulation_condition(frame_alpha, baseline_by_phase, srate,
                                             f"phase_locked_phase_modulation_stimulation_off_control|{subject}")

    # ---- benefit-prediction input: OFF-block deviation in the SAME frozen frame, for the displacement outcome ----
    off_latent_mean = _project_mean_latent(baseline_pooled, frame_alpha, srate)
    deviation_off_frame = rate_free_state_deviation(off_latent_mean)
    mean_off_frame = float(np.nanmean(deviation_off_frame))
    displacement = (phase_modulation_primary_result["mean_deviation"] - mean_off_frame) if phase_modulation_primary_result.get("status") == "computed" else None

    # ---- phase-modulation test, control-band control: own frame, own stimulation-ON trials ----
    try:
        no_stim_cb, stim_cb, qc_cb = _preprocess_fixed_band(subject, group, *ARTIFACT_CONTROL_BAND_HZ)
        baseline_pooled_cb = _retention_trials(no_stim_cb)[0]
        stim_by_phase_cb = _retention_trials(stim_cb, codes=list(PHASE_CONDITIONS))
        srate_cb = float(no_stim_cb.info["sfreq"])
        if baseline_pooled_cb.shape[0] >= MIN_BASELINE_TRIALS:
            frame_cb = _fit_frozen_frame(baseline_pooled_cb, srate_cb)
            phase_modulation_off_band_control_result = _phase_modulation_condition(frame_cb, stim_by_phase_cb, srate_cb,
                                                 f"phase_locked_phase_modulation_off_band_control|{subject}")
        else:
            phase_modulation_off_band_control_result = {"status": "refused", "reason": "too_few_control_band_baseline_trials"}
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        phase_modulation_off_band_control_result = {"status": "refused", "reason": f"{type(error).__name__}:{error}"}

    # ---- benefit-prediction input: this corpus's own behavioural accuracy-by-phase modulation depth ----
    rng_beh = np.random.default_rng(stable_seed(f"phase_locked_behaviour_modulation|{subject}"))
    behaviour_modulation = modulation_from_outcomes(trial_outcomes(subject), rng_beh)

    # ---- Reachability: circular distance between phase-minimising-component and phase-maximising-accuracy
    circular_gap = None
    if phase_modulation_primary_result.get("status") == "computed" and behaviour_modulation.get("optimal_phase_deg") is not None:
        phase_min_component = (phase_modulation_primary_result["harmonic"]["optimal_phase_deg"] + 180.0) % 360.0
        circular_gap = circular_distance_deg(phase_min_component, behaviour_modulation["optimal_phase_deg"])

    return {
        "status": "computed", "subject": subject, "group": group,
        "preprocessing_qc": qc, "n_baseline_trials": int(baseline_pooled.shape[0]),
        "phase_trial_counts_baseline": {str(c): int(len(baseline_by_phase.get(c, []))) for c in PHASE_CONDITIONS},
        "phase_trial_counts_stim": {str(c): int(len(stim_by_phase.get(c, []))) for c in PHASE_CONDITIONS},
        "presence_test": presence_record, "behaviour_link": behaviour_link_record,
        "phase_modulation_primary": phase_modulation_primary_result, "phase_modulation_off_band_control": phase_modulation_off_band_control_result,
        "phase_modulation_stimulation_off_control": phase_modulation_stimulation_off_control_result,
        "component_displacement_between_blocks": displacement, "behavioural_benefit_modulation": behaviour_modulation,
        "circular_distance_component_min_to_accuracy_max_deg": circular_gap,
    }


# ── Checkpointing (atomic, per participant) ────────────────────────────────────────────────────────────

def _checkpoint_path(participant: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", participant)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(participant: str) -> dict | None:
    path = _checkpoint_path(participant)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    record = data["record"]
    if record.get("status") == "refused" and str(record.get("reason", "")).startswith("exception:"):
        # An uncaught exception is a resource/implementation failure, not a reproducible fact about the
        # participant's data (unlike a deterministic refusal such as too_few_baseline_trials): trusting a
        # stale exception forever would let one crash silently outlive the bug that caused it. Retry.
        return None
    return record


def save_checkpoint(participant: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(participant)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def _run_and_checkpoint(subject: str, group: str) -> tuple[str, dict]:
    cached = load_checkpoint(subject)
    if cached is not None:
        return subject, cached
    try:
        record = process_participant(subject, group)
    except Exception as error:  # noqa: BLE001 -- one participant's crash must not lose the others
        record = {"status": "refused", "subject": subject, "group": group,
                   "reason": f"exception:{type(error).__name__}:{error}"}
    save_checkpoint(subject, record)
    return subject, record


# ── Group-level synthesis ───────────────────────────────────────────────────────────────────────────────

def _classify_presence(pooled: dict, reference_abs: float | None) -> str:
    if pooled["status"] == "underpowered_by_construction":
        return "inconclusive_below_detection_floor"
    if pooled["significant"]:
        return "the_component_is_present_in_the_non_invasive_timing_corpus"
    if reference_abs is not None and pooled["mdd"] is not None and pooled["mdd"] < reference_abs:
        return "the_component_is_absent_in_the_non_invasive_timing_corpus"
    return "inconclusive_below_detection_floor"


def _classify_behaviour_link(primary: dict, bias: dict) -> str:
    same_sign_significant = (
        primary.get("status") == "tested" and primary.get("significant")
        and bias.get("status") == "tested" and bias.get("significant")
        and np.sign(primary["mean_value"]) == np.sign(bias["mean_value"])
    )
    if same_sign_significant:
        return "component_behaviour_link_not_separable_from_a_participant_level_offset"
    if primary["status"] == "underpowered_by_construction":
        return "inconclusive_below_detection_floor"
    if primary["significant"]:
        return "the_component_predicts_accuracy_in_a_non_invasive_human_maintenance_delay"
    if primary["mdd"] is not None and primary["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return "the_component_does_not_predict_accuracy_in_a_non_invasive_human_maintenance_delay"
    return "inconclusive_below_detection_floor"


def _classify_phase_modulation(active_pop: dict | None, control_pop: dict | None, between: dict | None,
                       mdd_active: dict, behavioural_reference_amplitude: float | None,
                       control_band_active_sig: bool, baseline_control_active_sig: bool) -> str:
    active_sig = bool(active_pop and active_pop.get("circular_rotation_p_value") is not None
                       and active_pop["circular_rotation_p_value"] <= 0.05)
    control_sig = bool(control_pop and control_pop.get("circular_rotation_p_value") is not None
                        and control_pop["circular_rotation_p_value"] <= 0.05)
    between_sig = bool(between and between.get("participant_label_permutation_p_value") is not None
                        and between["participant_label_permutation_p_value"] <= 0.05)
    controls_reproduce = control_band_active_sig or baseline_control_active_sig
    if active_sig and control_sig:
        return "phase_modulation_is_present_in_both_groups"
    if active_sig and between_sig and not controls_reproduce:
        return "the_stimulation_phase_modulates_the_component_in_the_targeted_group_only"
    mdd_val = mdd_active.get("mdd") if mdd_active.get("status") == "computed" else None
    if not active_sig and mdd_val is not None and behavioural_reference_amplitude is not None \
            and mdd_val < behavioural_reference_amplitude:
        return "the_stimulation_phase_does_not_modulate_the_component"
    return "inconclusive_below_detection_floor"


def _classify_benefit_prediction(test: dict) -> str:
    if test.get("status") != "computed":
        return "underpowered_to_ask"
    if test["p_value"] <= 0.05:
        return "the_predictor_predicts_the_outcome_across_participants"
    mdd = minimum_detectable_correlation(test["n"])
    if mdd.get("status") == "computed" and mdd["mdd_r"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return "no_link_above_the_reported_bound"
    return "underpowered_to_ask"


def _group_participant_scalar(records: dict[str, dict], group: str, block_key: str, field: str) -> dict[str, float]:
    out = {}
    for subject, record in records.items():
        if record.get("group") != group or record.get("status") != "computed":
            continue
        value = record.get(block_key, {}).get(field)
        if value is not None and np.isfinite(value):
            out[subject] = float(value)
    return out


def benefit_prediction_cell(predictor: dict[str, float], outcome: dict[str, float], seed_tag: str) -> dict:
    shared = sorted(set(predictor) & set(outcome))
    if len(shared) < 4:
        return {"status": "too_few_participants", "n": len(shared)}
    x = np.array([predictor[s] for s in shared])
    y = np.array([outcome[s] for s in shared])
    rng = np.random.default_rng(stable_seed(seed_tag))
    result = partial_correlation_permutation_test(y, x, controls=[], n_perm=5000, rng=rng)
    result["branch"] = _classify_benefit_prediction(result)
    if result.get("status") == "computed":
        result["mdd"] = minimum_detectable_correlation(result["n"])
    return result


REFUSED_IMPLEMENTATION_FAILURE = "refused_implementation_failure"  # never a pre-declared branch name


def _exception_refusals(records: dict, participants: list) -> list[dict]:
    """Participants whose per-participant pipeline raised an uncaught exception (see
    _run_and_checkpoint). An exception is a resource/implementation failure, never a statement about the
    data. A test with zero contributing participants must consult this list before classifying: if it is
    non-empty, the crash -- not the data -- is why the test has nothing to say, and that must be named
    rather than folded into any pre-declared branch."""
    return [{"subject": s, "group": g, "reason": records[s].get("reason")}
            for s, g in participants
            if records[s].get("status") == "refused"
            and str(records[s].get("reason", "")).startswith("exception:")]


def _classify_or_refuse(n_contributing: int, exception_refusals: list[dict], classify_fn):
    """Calls classify_fn() to obtain a pre-declared branch name, UNLESS the test has zero contributing
    participants and at least one participant failed with an uncaught exception. A branch may only fire
    on data; when the zero is a crash rather than a data outcome, no pre-declared branch name --
    including 'inconclusive_below_detection_floor' -- is permitted, and REFUSED_IMPLEMENTATION_FAILURE is
    returned instead."""
    if n_contributing == 0 and exception_refusals:
        return REFUSED_IMPLEMENTATION_FAILURE
    return classify_fn()


def synthesize_output(records: dict, participants: list, active: list, control: list, gate: dict,
                       t0: float) -> dict:
    """Turns per-participant records (each already 'computed' or 'refused' -- see process_participant and
    _run_and_checkpoint) into the artifact's group-level tests and branches. Kept separate from main() so
    a synthetic all-refused-by-exception `records` bundle can be run through it directly in tests, with no
    EEG data and no subprocess pool required."""
    n_seen = len(participants)
    n_computed = sum(1 for r in records.values() if r.get("status") == "computed")
    n_refused = n_seen - n_computed
    n_active_seen, n_control_seen = len(active), len(control)
    n_active_computed = sum(1 for s, g in participants if g == "active" and records[s].get("status") == "computed")
    n_control_computed = sum(1 for s, g in participants if g == "control" and records[s].get("status") == "computed")

    exception_refusals = _exception_refusals(records, participants)

    # ---- presence test: pooled across all computed participants, both groups ----
    per_participant_a = {s: r["presence_test"]["gate"]["signed_effect"] for s, r in records.items()
                          if r.get("status") == "computed" and r["presence_test"]["gate"].get("status") == "computed"}
    pooled_a = _patient_clustered_test(per_participant_a)
    scalp_reference = recording_tier_scalp_reference()
    branch_a = _classify_or_refuse(len(per_participant_a), exception_refusals,
                                    lambda: _classify_presence(pooled_a, scalp_reference["abs_mean_value"]))

    # ---- behaviour-link test: pooled across all computed participants with a computed behaviour cell ----
    per_participant_b_primary = {s: r["behaviour_link"]["primary"]["r"] for s, r in records.items()
                                  if r.get("status") == "computed" and r["behaviour_link"].get("status") == "computed"
                                  and r["behaviour_link"]["primary"].get("status") == "computed"}
    per_participant_b_bias = {s: r["behaviour_link"]["bias_only"]["r"] for s, r in records.items()
                               if r.get("status") == "computed" and r["behaviour_link"].get("status") == "computed"
                               and r["behaviour_link"]["bias_only"].get("status") == "computed"}
    pooled_b_primary = _patient_clustered_test(per_participant_b_primary)
    pooled_b_bias = _patient_clustered_test(per_participant_b_bias)
    branch_b = _classify_or_refuse(len(per_participant_b_primary), exception_refusals,
                                    lambda: _classify_behaviour_link(pooled_b_primary, pooled_b_bias))
    total_errors = sum(r["behaviour_link"].get("n_errors", 0) for r in records.values() if r.get("status") == "computed")
    human_null_context = three_corpus_human_null_context()

    # ---- phase-modulation and benefit-prediction tests, gated on the presence test ----
    _not_run_reason = ("presence_test_did_not_run_on_real_data" if branch_a == REFUSED_IMPLEMENTATION_FAILURE
                        else "presence_branch_is_not_the_present_branch")
    phase_modulation = {"status": "not_run", "reason": _not_run_reason, "presence_branch": branch_a}
    benefit_prediction = {"status": "not_run", "reason": _not_run_reason, "presence_branch": branch_a}
    if branch_a == "the_component_is_present_in_the_non_invasive_timing_corpus":
        def _population(group_name: str, block_key: str) -> dict | None:
            rows = [{"harmonic": records[s][block_key]["harmonic"]} for s, g in participants
                    if g == group_name and records[s].get("status") == "computed"
                    and records[s][block_key].get("status") == "computed"]
            return group_vector_test(rows, ("harmonic",), f"phase_locked_population|{group_name}|{block_key}",
                                       n_perm=N_PERM_PHASE_MODULATION)

        active_primary_pop = _population("active", "phase_modulation_primary")
        control_primary_pop = _population("control", "phase_modulation_primary")
        active_control_band_pop = _population("active", "phase_modulation_off_band_control")
        active_baseline_control_pop = _population("active", "phase_modulation_stimulation_off_control")

        between_rows = [{"group": g, "primary_harmonic": records[s]["phase_modulation_primary"]["harmonic"]}
                         for s, g in participants if records[s].get("status") == "computed"
                         and records[s]["phase_modulation_primary"].get("status") == "computed"]
        between_group = active_control_difference(between_rows, "primary_harmonic", N_PERM_PHASE_MODULATION)

        active_amplitudes = _group_participant_scalar(records, "active", "phase_modulation_primary", None) \
            if False else {s: records[s]["phase_modulation_primary"]["harmonic"]["amplitude"] for s, g in participants
                            if g == "active" and records[s].get("status") == "computed"
                            and records[s]["phase_modulation_primary"].get("status") == "computed"}
        control_amplitudes = {s: records[s]["phase_modulation_primary"]["harmonic"]["amplitude"] for s, g in participants
                               if g == "control" and records[s].get("status") == "computed"
                               and records[s]["phase_modulation_primary"].get("status") == "computed"}
        mdd_active_one_sample = minimum_detectable_paired_difference(list(active_amplitudes.values())) \
            if len(active_amplitudes) >= 2 else {"status": "not_computable"}
        between_group_mdd = minimum_detectable_group_difference(
            len(active_amplitudes), len(control_amplitudes),
            pooled_sd(list(active_amplitudes.values()), list(control_amplitudes.values())))

        rows_behaviour_active = [{"harmonic_log_odds": records[s]["behavioural_benefit_modulation"]}
                                  for s, g in participants if g == "active" and records[s].get("status") == "computed"
                                  and records[s]["behavioural_benefit_modulation"].get("depth") is not None]
        behavioural_pop_active = None
        for row in rows_behaviour_active:
            row["harmonic_log_odds"] = {
                "cosine": row["harmonic_log_odds"]["depth"] * np.cos(np.deg2rad(row["harmonic_log_odds"]["optimal_phase_deg"])),
                "sine": row["harmonic_log_odds"]["depth"] * np.sin(np.deg2rad(row["harmonic_log_odds"]["optimal_phase_deg"])),
            }
        if len(rows_behaviour_active) >= 3:
            behavioural_pop_active = group_vector_test(rows_behaviour_active, ("harmonic_log_odds",),
                                                          "phase_locked_behaviour_link_population_active", N_PERM_PHASE_MODULATION)

        control_band_active_sig = bool(active_control_band_pop
                                        and active_control_band_pop.get("circular_rotation_p_value") is not None
                                        and active_control_band_pop["circular_rotation_p_value"] <= 0.05)
        baseline_control_active_sig = bool(active_baseline_control_pop
                                            and active_baseline_control_pop.get("circular_rotation_p_value") is not None
                                            and active_baseline_control_pop["circular_rotation_p_value"] <= 0.05)
        branch_c = _classify_phase_modulation(
            active_primary_pop, control_primary_pop, between_group, mdd_active_one_sample,
            behavioural_pop_active.get("population_amplitude") if behavioural_pop_active else None,
            control_band_active_sig, baseline_control_active_sig)

        circular_gaps = [r["circular_distance_component_min_to_accuracy_max_deg"] for r in records.values()
                          if r.get("status") == "computed"
                          and r.get("circular_distance_component_min_to_accuracy_max_deg") is not None]
        circular_gap_summary = None
        if len(circular_gaps) >= 3:
            arr = np.array(circular_gaps)
            rng_cd = np.random.default_rng(stable_seed("phase_locked_circular_gap_bootstrap"))
            mean_val, lower, upper = bootstrap_ci(arr, lambda d: float(np.mean(d)), rng=rng_cd, n_boot=5000)
            circular_gap_summary = {"n": len(circular_gaps), "mean_deg": mean_val, "ci_lower_deg": lower,
                                     "ci_upper_deg": upper}

        phase_modulation = {
            "decision_rule": PHASE_MODULATION_DECISION_RULE, "branch": branch_c,
            "active_group_population_test": active_primary_pop,
            "control_group_population_test": control_primary_pop,
            "between_group_test": between_group, "between_group_minimum_detectable_difference": between_group_mdd,
            "artifact_control_band_hz": list(ARTIFACT_CONTROL_BAND_HZ),
            "artifact_control_band_active_group_population_test": active_control_band_pop,
            "artifact_control_band_active_group_significant": control_band_active_sig,
            "baseline_block_control_active_group_population_test": active_baseline_control_pop,
            "baseline_block_control_active_group_significant": baseline_control_active_sig,
            "active_group_minimum_detectable_amplitude_one_sample": mdd_active_one_sample,
            "behavioural_accuracy_by_phase_population_amplitude_active_group": behavioural_pop_active,
            "n_active_contributing": len(active_amplitudes), "n_control_contributing": len(control_amplitudes),
            "circular_distance_component_minimum_to_accuracy_maximum": circular_gap_summary,
        }

        # ---- benefit-prediction test ----
        central_active = _group_predictor(records, participants, "active", "presence_test", "median_deviation")
        spread_active = _group_predictor(records, participants, "active", "presence_test", "iqr_deviation", is_iqr=True)
        presence_active = {s: records[s]["presence_test"]["gate"]["signed_effect"] for s, g in participants
                            if g == "active" and records[s].get("status") == "computed"
                            and records[s]["presence_test"]["gate"].get("status") == "computed"}
        central_control = _group_predictor(records, participants, "control", "presence_test", "median_deviation")
        spread_control = _group_predictor(records, participants, "control", "presence_test", "iqr_deviation", is_iqr=True)
        presence_control = {s: records[s]["presence_test"]["gate"]["signed_effect"] for s, g in participants
                             if g == "control" and records[s].get("status") == "computed"
                             and records[s]["presence_test"]["gate"].get("status") == "computed"}

        outcome_benefit_active = {s: records[s]["behavioural_benefit_modulation"]["depth"] for s, g in participants
                                   if g == "active" and records[s].get("status") == "computed"
                                   and records[s]["behavioural_benefit_modulation"].get("depth") is not None}
        outcome_displacement_active = {s: records[s]["component_displacement_between_blocks"] for s, g in participants
                                        if g == "active" and records[s].get("status") == "computed"
                                        and records[s].get("component_displacement_between_blocks") is not None}
        outcome_benefit_control = {s: records[s]["behavioural_benefit_modulation"]["depth"] for s, g in participants
                                    if g == "control" and records[s].get("status") == "computed"
                                    and records[s]["behavioural_benefit_modulation"].get("depth") is not None}
        outcome_displacement_control = {s: records[s]["component_displacement_between_blocks"] for s, g in participants
                                         if g == "control" and records[s].get("status") == "computed"
                                         and records[s].get("component_displacement_between_blocks") is not None}

        benefit_prediction = {"decision_rule": BENEFIT_PREDICTION_DECISION_RULE, "active": {}, "control": {}}
        for group_name, central, spread, presence, benefit, displ in (
                ("active", central_active, spread_active, presence_active, outcome_benefit_active, outcome_displacement_active),
                ("control", central_control, spread_control, presence_control, outcome_benefit_control, outcome_displacement_control)):
            benefit_prediction[group_name] = {
                "central_value_vs_behavioural_benefit": benefit_prediction_cell(central, benefit, f"phase_locked_d|{group_name}|central|benefit"),
                "central_value_vs_displacement": benefit_prediction_cell(central, displ, f"phase_locked_d|{group_name}|central|displacement"),
                "spread_vs_behavioural_benefit": benefit_prediction_cell(spread, benefit, f"phase_locked_d|{group_name}|spread|benefit"),
                "spread_vs_displacement": benefit_prediction_cell(spread, displ, f"phase_locked_d|{group_name}|spread|displacement"),
                "presence_statistic_vs_behavioural_benefit": benefit_prediction_cell(presence, benefit, f"phase_locked_d|{group_name}|presence|benefit"),
                "presence_statistic_vs_displacement": benefit_prediction_cell(presence, displ, f"phase_locked_d|{group_name}|presence|displacement"),
            }

    mandatory_test_blocked_by_exception = REFUSED_IMPLEMENTATION_FAILURE in (branch_a, branch_b)
    top_status = ("incomplete_mandatory_test_blocked_by_exception" if mandatory_test_blocked_by_exception
                  else "complete")

    output = {
        "schema_version": "1.0.0", "analysis_id": "phase_locked_scalp_stimulation_component",
        "status": top_status, "code_commit": git_commit(ROOT),
        "participant_exception_refusals": exception_refusals,
        "scope": {
            "corpus": "closed-loop, phase-tuned alternating-current stimulation to the scalp during a "
                      "working-memory maintenance delay, locked to each participant's own endogenous 8-14 "
                      "Hz oscillation at six evenly spaced phase lags, with an active group stimulated over "
                      "the recorded alpha source and a control group stimulated away from it",
            "stimulation_band_hz": list(STIMULATION_BAND_HZ), "artifact_control_band_hz": list(ARTIFACT_CONTROL_BAND_HZ),
            "n_active_seen": n_active_seen, "n_control_seen": n_control_seen,
            "n_active_computed": n_active_computed, "n_control_computed": n_control_computed,
            "n_seen": n_seen, "n_computed": n_computed, "n_refused": n_refused,
            "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
            "n_permutations_phase_modulation": N_PERM_PHASE_MODULATION, "n_rotation_null_draws_presence": N_ROTATION_NULL_DRAWS,
        },
        "decision_rules": {"presence_test": PRESENCE_DECISION_RULE, "behaviour_link": BEHAVIOUR_LINK_DECISION_RULE,
                            "phase_modulation": PHASE_MODULATION_DECISION_RULE, "benefit_prediction": BENEFIT_PREDICTION_DECISION_RULE},
        "reproduction_gate": gate,
        "recording_tier_scalp_reference": scalp_reference,
        "three_corpus_human_maintenance_null_context": human_null_context,
        "participant_status": {s: {"group": g, "status": records[s].get("status"),
                                     "reason": records[s].get("reason")} for s, g in participants},
        "presence_test": {"decision_rule": PRESENCE_DECISION_RULE, "branch": branch_a, "pooled_patient_test": pooled_a,
                     "reference_effect_scalp_tier": scalp_reference, "n_participants_contributing": len(per_participant_a)},
        "behaviour_link": {"decision_rule": BEHAVIOUR_LINK_DECISION_RULE, "branch": branch_b,
                     "primary_pooled_patient_test": pooled_b_primary, "bias_only_pooled_patient_test": pooled_b_bias,
                     "n_participants_contributing": len(per_participant_b_primary), "n_errors_total": int(total_errors),
                     "three_corpus_invasive_human_null_context": human_null_context},
        "phase_modulation": phase_modulation, "benefit_prediction": benefit_prediction,
        "scope_and_limitations": (
            "It can establish, in a human, non-invasively, whether the phase of an ongoing oscillation at "
            "the moment of stimulation delivery changes this project's rate-free state-deviation component, "
            "and if so at which phase. It cannot establish where the current went: this is a scalp montage "
            "with no measured field distribution, so nothing here licenses an anatomical target. It cannot "
            "establish an intensity: the energy-to-current conversion this project would need is "
            "uncalibrated, so only ratios would ever be reportable, and none are computed here. Its "
            "behavioural outcome is binary, so it cannot separate a swap from an imprecision."
        ),
        "wall_clock_s": time.time() - t0,
    }
    if branch_a == REFUSED_IMPLEMENTATION_FAILURE:
        output["presence_test"]["exception_refusals"] = exception_refusals
    if branch_b == REFUSED_IMPLEMENTATION_FAILURE:
        output["behaviour_link"]["exception_refusals"] = exception_refusals
    return output


def _group_predictor(records: dict, participants: list, group_name: str, block_key: str, field: str,
                      is_iqr: bool = False) -> dict[str, float]:
    out = {}
    for subject, group in participants:
        if group != group_name or records[subject].get("status") != "computed":
            continue
        value = records[subject][block_key].get(field)
        if value is None:
            continue
        out[subject] = float(value[1] - value[0]) if is_iqr else float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--active", type=int, default=None)
    parser.add_argument("--control", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    gate = reproduction_gate()
    if gate["status"] != "matched":
        output = {"schema_version": "1.0.0", "analysis_id": "phase_locked_scalp_stimulation_component",
                   "status": "void_reproduction_gate_did_not_reproduce", "reproduction_gate": gate,
                   "code_commit": git_commit(ROOT), "wall_clock_s": time.time() - t0}
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(canonical_json(output))
        print(json.dumps({"status": output["status"], "max_abs_diff": gate["max_abs_diff"]}, indent=2))
        return

    active = ACTIVE_SUBJECTS[:args.active] if args.active is not None else list(ACTIVE_SUBJECTS)
    control = CONTROL_SUBJECTS[:args.control] if args.control is not None else list(CONTROL_SUBJECTS)
    participants = [(s, "active") for s in active] + [(s, "control") for s in control]

    records: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=max(1, min(3, args.workers))) as executor:
        futures = {executor.submit(_run_and_checkpoint, subject, group): subject
                   for subject, group in participants}
        for future in as_completed(futures):
            subject, record = future.result()
            records[subject] = record
            print(f"done {subject} ({record.get('group')}): {record.get('status')}"
                  f"{'' if record.get('status') == 'computed' else ' -- ' + str(record.get('reason'))}",
                  flush=True)

    output = synthesize_output(records, participants, active, control, gate, t0)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(output))
    phase_modulation = output.get("phase_modulation")
    print(json.dumps({
        "status": output["status"], "reproduction_gate": gate["status"],
        "n_computed": output["scope"]["n_computed"], "n_refused": output["scope"]["n_refused"],
        "presence_branch": output["presence_test"]["branch"], "behaviour_link_branch": output["behaviour_link"]["branch"],
        "phase_modulation_branch": phase_modulation.get("branch", phase_modulation.get("status")),
        "wall_clock_s": output["wall_clock_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
