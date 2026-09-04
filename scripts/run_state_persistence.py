"""run_state_persistence.py -- census of r(gap), the across-trial correlation
of a population's leading latent position between two equal-length
sub-windows of an epoch, as a function of the gap between the windows.

This observable replaces the lagged-autocovariance decay fit
(``variance_partition.py``'s ``slow``/``static`` split) as the vehicle for
any claim about the timescale of a human working-memory delay's
trial-specific state: it never removes a trial's own temporal mean, so
unlike that decay fit it cannot book a slow component as a static offset
(see ``results/variance_partition.json``'s ``flow_dissociation_void``
key for the argument). ``variance_partition.py`` and its artifact are left
exactly as they were; this module and this script add a new artifact beside
them rather than replacing anything.

Every row carries its own per-session Poisson negative control
(``state_persistence.poisson_null_r_gap_profile``): a rate-matched
population with an identical time course on every simulated trial, so no
trial-specific structure exists in it by construction.

Scope actually run (a documented narrowing of the full cross-product of
epoch x bin width x window mode x window count, for wall-clock reasons):
  - the PRIMARY grid runs every epoch, both bin widths, each corpus's own
    native window, and 4 sub-windows, for every (dataset, session,
    structure) reachable through corpus_sessions.iter_all_corpora, plus the
    matched mouse-ALM comparison and matched-sensitivity arm;
  - the delay epoch additionally gets the common 2.3 s floor window (both
    bin widths) and 3 sub-windows (both bin widths, native window), because
    delay is where the deciding contrast and the anatomical comparison live;
  - encoding and delay, pooled structure, 100 ms, native window, feed the
    within-session maintenance-versus-encoding discriminator;
  - the macaque prefrontal delay-period population (Panichello et al. 2024)
    is run through the identical estimator, pooled per session, as the
    second discriminator.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from corpus_sessions import (  # noqa: E402
    ALM_WINDOW_S, EPOCH_WINDOWS_S, alm_data_directory, data_root, iter_alm, iter_all_corpora,
    load_alm_raw_session,
)
from spike_pipeline import build_psth  # noqa: E402
from statistics import paired_sign_flip_test, spearman_permutation_test, stable_seed  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from state_persistence import (  # noqa: E402
    attenuation_constancy_check, binomial_thin, calibration_ladder, classify_gap_profile, classify_lag_profile,
    firing_rate_hz, paired_vs_null_contrast, per_unit_permutation_null_r_gap_profile,
    per_unit_permutation_null_r_lag_profile, permutation_null_validation, poisson_null_r_gap_profile,
    poisson_null_r_lag_profile, patient_clustered_lag_inference, r_gap_profile, r_lag_profile, reachability_note,
)

EPOCHS = ("baseline", "encoding", "delay", "probe")
BIN_WIDTHS_MS = (100.0, 200.0)
N_SPLITS = 12
N_NULL_REPLICATES = 20
NULL_SPLITS_PER_REPLICATE = 6
MATCHED_POWER_DRAWS = 10
MATCHED_POWER_N_SPLITS = 8
POWER_RESAMPLES = 300
HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")
PANICHELLO_DELAY_WINDOW_MS = (300.0, 1450.0)  # matches scripts/run_panichello_pipeline.py
# The macaque prefrontal window-count-profile arm's split and permutation-replicate counts are lighter
# than the human/ALM module defaults (N_SPLITS, N_NULL_REPLICATES above) because each session carries a
# large simultaneous population and a single fit already costs several seconds. Named as explicit
# parameters (default to these constants) rather than bare literals so a caller who needs the module
# default for a direct cross-arm comparison can request it, and so every row this arm contributes
# carries its own settings via the standard n_splits_requested/n_replicates_requested fields -- see
# scripts/run_persistence_estimator_split_count_sensitivity.py, which calls this arm at both regimes to
# measure whether the lighter setting itself, not the species, produces this arm's distinctive slope.
PANICHELLO_PROFILE_N_SPLITS = 10
PANICHELLO_PROFILE_N_NULL_REPLICATES = 10
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence.json"
VARIANCE_PARTITION_PATH = Path(__file__).resolve().parents[1] / "results" / "variance_partition.json"


def _seed(*parts) -> int:
    return stable_seed("|".join(str(p) for p in parts))


def _counts_from_spikes(spike_lists, onset, window_s: float, bin_ms: float) -> np.ndarray:
    rate = build_psth(spike_lists, onset, bin_ms=bin_ms, smooth_ms=0.0, window_s=window_s)
    return rate * (bin_ms / 1000.0)


def _run_row(counts: np.ndarray, n_windows: int, seed: int,
             n_splits: int = N_SPLITS, n_null_replicates: int = N_NULL_REPLICATES) -> dict | None:
    if counts.shape[0] < 8 or counts.shape[2] // n_windows < 1:
        return None
    profile = r_gap_profile(counts, n_splits=n_splits, rng=np.random.default_rng(seed), n_windows=n_windows)
    if profile["status"] != "fitted":
        return {"profile": profile, "null": None, "n_trials": int(counts.shape[0]),
                "n_units": int(counts.shape[1]), "n_bins": int(counts.shape[2])}
    null = poisson_null_r_gap_profile(
        counts, n_replicates=n_null_replicates, n_splits_per_replicate=NULL_SPLITS_PER_REPLICATE,
        rng=np.random.default_rng(seed + 1), n_windows=n_windows)
    return {"profile": profile, "null": null, "n_trials": int(counts.shape[0]),
            "n_units": int(counts.shape[1]), "n_bins": int(counts.shape[2])}


def human_primary_rows(root: Path) -> list[dict]:
    """Every epoch, both bin widths, each corpus's own native window, 4
    sub-windows -- mirrors variance_partition.py's Phase-1 grid exactly."""
    rows = []
    t0 = time.time()
    for i, meta in enumerate(iter_all_corpora(root)):
        for epoch in EPOCHS:
            window_s = meta["epoch_windows"][epoch]
            onset = meta["epoch_onsets"][epoch]
            for bin_ms in BIN_WIDTHS_MS:
                counts = _counts_from_spikes(meta["spike_lists"], onset, window_s, bin_ms)
                seed = _seed(meta["dataset"], meta["session"], meta["structure"], epoch, bin_ms, "w4", "native")
                run = _run_row(counts, 4, seed)
                if run is None:
                    continue
                rows.append({
                    "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                    "structure": meta["structure"], "epoch": epoch, "bin_ms": bin_ms,
                    "n_windows": 4, "window_mode": "native", "window_s": window_s, **run,
                })
        if (i + 1) % 10 == 0:
            print(f"  primary grid: session {i + 1}, {len(rows)} rows so far, "
                  f"{time.time() - t0:.1f}s elapsed", file=sys.stderr, flush=True)
    print(f"  primary grid done: {len(rows)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return rows


def human_delay_secondary_rows(root: Path) -> list[dict]:
    """Delay epoch only: the common 2.3 s floor window (both bin widths, 4
    sub-windows) plus 3 sub-windows at the native window (both bin widths).
    Narrower than the full cross-product of every epoch at both window
    modes and both window counts; restricted to the deciding epoch for
    wall-clock reasons, stated here rather than run silently -- see the
    artifact's ``scope`` field."""
    rows = []
    t0 = time.time()
    for meta in iter_all_corpora(root):
        onset = meta["epoch_onsets"]["delay"]
        native_window_s = meta["epoch_windows"]["delay"]
        floor_window_s = EPOCH_WINDOWS_S["delay"]
        for bin_ms in BIN_WIDTHS_MS:
            if floor_window_s != native_window_s:
                counts = _counts_from_spikes(meta["spike_lists"], onset, floor_window_s, bin_ms)
                seed = _seed(meta["dataset"], meta["session"], meta["structure"], "delay", bin_ms, "w4", "floor")
                run = _run_row(counts, 4, seed)
                if run is not None:
                    rows.append({
                        "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                        "structure": meta["structure"], "epoch": "delay", "bin_ms": bin_ms,
                        "n_windows": 4, "window_mode": "floor", "window_s": floor_window_s, **run,
                    })
            counts = _counts_from_spikes(meta["spike_lists"], onset, native_window_s, bin_ms)
            seed = _seed(meta["dataset"], meta["session"], meta["structure"], "delay", bin_ms, "w3", "native")
            run = _run_row(counts, 3, seed)
            if run is not None:
                rows.append({
                    "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                    "structure": meta["structure"], "epoch": "delay", "bin_ms": bin_ms,
                    "n_windows": 3, "window_mode": "native", "window_s": native_window_s, **run,
                })
    print(f"  delay secondary grid done: {len(rows)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return rows


def alm_census_rows(root: Path) -> list[dict]:
    rows = []
    for bin_ms in BIN_WIDTHS_MS:
        for meta in iter_alm(root, bin_ms=bin_ms, window_s=ALM_WINDOW_S):
            seed = _seed(meta["dataset"], meta["session"], "delay", bin_ms, "w4", "native")
            run = _run_row(meta["counts"], 4, seed)
            if run is None:
                continue
            rows.append({
                "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                "structure": "pooled", "epoch": "delay", "bin_ms": bin_ms,
                "n_windows": 4, "window_mode": "native", "window_s": ALM_WINDOW_S, **run,
            })
    return rows


def matched_sensitivity_alm(root: Path, human_median_units: int, human_median_trials: int,
                             human_median_rate_hz: float) -> dict:
    """Degrade ALM to the human effect size by unit count, trial count, and
    firing-rate binomial thinning -- matching only on
    units and trials is not matching on the thing that determines whether a
    decay is detectable, since human r(1) and ALM r(1) differ far more than
    a unit/trial-count difference alone would predict. Reports a thinning
    ladder, and the fraction of resamples at the human session count in
    which the gap effect (r1 - r3) is detected at 0.05 -- the power
    statement any null result elsewhere in this artifact is read against."""
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return {"status": "not_available", "reason": "ALM data directory not staged"}

    sessions = []
    alm_rates = []
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=100.0, window_s=ALM_WINDOW_S)
        if session is None:
            continue
        sessions.append(session["control_counts"])
        alm_rates.append(firing_rate_hz(session["control_counts"], ALM_WINDOW_S))
    if not sessions:
        return {"status": "not_available", "reason": "no eligible ALM sessions"}

    alm_median_rate = float(np.median(alm_rates))
    rate_matched_keep_probability = min(1.0, human_median_rate_hz / alm_median_rate) if alm_median_rate > 0 else 1.0
    thinning_ladder = sorted({1.0, round(rate_matched_keep_probability, 4), 0.10, 0.05}, reverse=True)

    regimes = {}
    for keep_probability in thinning_ladder:
        per_session_r1, per_session_r3 = [], []
        for i, counts in enumerate(sessions):
            rng = np.random.default_rng(_seed("matched_sensitivity", i, keep_probability))
            draws = {1: [], 3: []}
            for _ in range(MATCHED_POWER_DRAWS):
                draw = counts
                if human_median_units < draw.shape[1]:
                    unit_idx = rng.choice(draw.shape[1], size=human_median_units, replace=False)
                    draw = draw[:, unit_idx, :]
                if human_median_trials < draw.shape[0]:
                    trial_idx = rng.choice(draw.shape[0], size=human_median_trials, replace=False)
                    draw = draw[trial_idx]
                draw = binomial_thin(draw, keep_probability, rng)
                profile = r_gap_profile(draw, n_splits=MATCHED_POWER_N_SPLITS, rng=rng, n_windows=4)
                if profile["status"] != "fitted" or 1 not in profile["gaps"] or 3 not in profile["gaps"]:
                    continue
                draws[1].append(profile["gaps"][1]["r_median"])
                draws[3].append(profile["gaps"][3]["r_median"])
            if draws[1] and draws[3]:
                per_session_r1.append(float(np.median(draws[1])))
                per_session_r3.append(float(np.median(draws[3])))
        r1 = np.array(per_session_r1)
        r3 = np.array(per_session_r3)
        n = len(r1)
        if n < 4:
            regimes[keep_probability] = {"status": "underpowered_by_construction", "n_sessions": n}
            continue
        test = paired_sign_flip_test(r1, r3, alternative="greater")
        achieved_rate = keep_probability * alm_median_rate  # binomial thinning is unbiased in expectation
        rng_power = np.random.default_rng(_seed("power_resample", keep_probability))
        n_resample = min(18, n)  # human delay session count reference (DANDI 000469, n=18)
        p_values = []
        for _ in range(POWER_RESAMPLES):
            idx = rng_power.choice(n, size=n_resample, replace=False)
            p_values.append(paired_sign_flip_test(r1[idx], r3[idx], alternative="greater", n_perm=2000, rng=rng_power)["p_value"])
        regimes[keep_probability] = {
            "status": "tested", "n_sessions": int(n), "median_rate_hz": achieved_rate,
            "r1_median": float(np.median(r1)), "r3_median": float(np.median(r3)),
            "n_r1_gt_r3": int((r1 > r3).sum()), "mean_diff": test["mean_diff"], "p_value": test["p_value"],
            "power_at_0p05_resampled_to_n18": float(np.mean(np.array(p_values) < 0.05)),
        }
    return {
        "status": "tested", "n_sessions": len(sessions), "human_median_units": human_median_units,
        "human_median_trials": human_median_trials, "human_median_rate_hz": human_median_rate_hz,
        "alm_median_rate_hz": alm_median_rate, "rate_matched_keep_probability": rate_matched_keep_probability,
        "regimes": {str(k): v for k, v in regimes.items()},
    }


def _panichello_directory(root: Path) -> Path | None:
    config = json.loads((Path(__file__).resolve().parents[1] / "config" / "datasets.json").read_text())
    entry = config["datasets"]["panichello_2024"]  # raise if the registry key is missing/mistyped, not a silent skip
    path = root / entry["local_path"]
    return path if path.is_dir() else None


def panichello_rows(root: Path, n_splits: int = PANICHELLO_PROFILE_N_SPLITS,
                     n_null_replicates: int = PANICHELLO_PROFILE_N_NULL_REPLICATES) -> dict:
    """Macaque lPFC delay-period population, pooled per session (no
    per-channel area label exists in the deposited .mat files -- see
    ``key_list_inspected`` below), run through the identical r(gap)
    estimator as the human corpora. ``n_splits``/``n_null_replicates``
    default to this arm's lighter settings (see PANICHELLO_PROFILE_N_SPLITS
    above) but are explicit parameters so this arm can be re-run at the
    human/ALM module default when a comparison needs matched settings
    rather than a documented, disclosed mismatch."""
    directory = _panichello_directory(root)
    if directory is None:
        return {"status": "not_available", "reason": "Panichello_2024 not staged", "rows": []}

    files = sorted(glob.glob(str(directory / "*.mat")))
    if not files:
        return {"status": "not_available", "reason": "no .mat files found", "rows": []}

    raw0 = loadmat(files[0], squeeze_me=True)
    key_list_inspected = sorted(k for k in raw0.keys() if not k.startswith("__"))

    rows = []
    for path in files:
        raw = loadmat(path, squeeze_me=True)
        spikes = np.asarray(raw["spks"], dtype=float)  # (trial, time, unit)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
        spikes = spikes[correct]
        for bin_ms in BIN_WIDTHS_MS:
            starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], bin_ms)
            binned = []
            for start in starts:
                mask = (time_ms >= start) & (time_ms < start + bin_ms)
                binned.append(spikes[:, mask, :].sum(axis=1))
            counts = np.stack(binned, axis=2)  # (trial, unit, bin)
            seed = _seed("panichello", Path(path).stem, bin_ms, "w4", "native")
            # Lighter null replication than the human/ALM grids by default (see
            # PANICHELLO_PROFILE_N_SPLITS/PANICHELLO_PROFILE_N_NULL_REPLICATES): these
            # sessions carry far more trials and units (a large simultaneous population),
            # so a single r_gap fit already costs several seconds. This arm's role in the
            # project's cross-species comparison has changed over time, so its settings are
            # not assumed harmless here -- any caller comparing this arm's numbers against
            # the human/ALM arms must report this arm's settings alongside them (see
            # ``estimator_settings`` below) rather than treating the module default as
            # having applied uniformly.
            run = _run_row(counts, 4, seed, n_splits=n_splits, n_null_replicates=n_null_replicates)
            if run is None:
                continue
            rows.append({
                "dataset": "panichello_2024", "patient": Path(path).stem, "session": Path(path).stem,
                "structure": "pooled", "epoch": "delay", "bin_ms": bin_ms,
                "n_windows": 4, "window_mode": "native", "window_s": (PANICHELLO_DELAY_WINDOW_MS[1] - PANICHELLO_DELAY_WINDOW_MS[0]) / 1000.0,
                **run,
            })
    return {"status": "tested", "key_list_inspected": key_list_inspected,
            "estimator_settings": {
                "n_splits": n_splits, "n_null_replicates": n_null_replicates,
                "null_splits_per_replicate": NULL_SPLITS_PER_REPLICATE,
                "differs_from_human_and_alm_module_default": (n_splits, n_null_replicates) != (N_SPLITS, N_NULL_REPLICATES),
                "human_and_alm_module_default": {"n_splits": N_SPLITS, "n_null_replicates": N_NULL_REPLICATES},
            },
            "label_convention_note": "No per-channel area/region field exists in the per-session .mat files "
                                      "(confirmed by the key list above); config/datasets.json's "
                                      "'monkey_cluster_area_inferred' convention (session date/monkey, not "
                                      "per-channel anatomy) is the finest labeling this deposit supports, which "
                                      "does not license a per-area split -- the pooled-session population below "
                                      "is what is estimable here.",
            "rows": rows}


PROFILE_WINDOW_COUNTS = (4, 6, 8)
PROFILE_EPOCHS = ("encoding", "delay")
PROFILE_BIN_MS = 100.0
PROFILE_N_SPLITS = 12
PROFILE_N_NULL_REPLICATES = 20
PROFILE_NULL_SPLITS_PER_REPLICATE = 6
PROFILE_MIN_BINS_PER_WINDOW = 2
PROFILE_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_profile.json"


def _profile_run_row(counts: np.ndarray, n_windows: int, seed: int) -> dict:
    """Every row carries a status even when it cannot be estimated -- a
    session that cannot be estimated is a row with a status, never a
    silent omission -- and both the Poisson and per-unit-permutation nulls
    beside the profile, since one cannot substitute for the other."""
    bins_per_window = counts.shape[2] // n_windows
    base = {
        "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
        "n_bins": int(counts.shape[2]), "bins_per_window": int(bins_per_window),
    }
    if counts.shape[0] < 8:
        return {**base, "profile": {"status": "fewer_than_eight_trials"}, "null_poisson": None, "null_permutation": None}
    if bins_per_window < PROFILE_MIN_BINS_PER_WINDOW:
        return {**base, "profile": {"status": "bins_per_window_below_floor"}, "null_poisson": None, "null_permutation": None}
    profile = r_gap_profile(counts, n_splits=PROFILE_N_SPLITS, rng=np.random.default_rng(seed), n_windows=n_windows)
    if profile["status"] != "fitted":
        return {**base, "profile": profile, "null_poisson": None, "null_permutation": None}
    null_poisson = poisson_null_r_gap_profile(
        counts, n_replicates=PROFILE_N_NULL_REPLICATES, n_splits_per_replicate=PROFILE_NULL_SPLITS_PER_REPLICATE,
        rng=np.random.default_rng(seed + 1), n_windows=n_windows)
    null_permutation = per_unit_permutation_null_r_gap_profile(
        counts, n_replicates=PROFILE_N_NULL_REPLICATES, n_splits_per_replicate=PROFILE_NULL_SPLITS_PER_REPLICATE,
        rng=np.random.default_rng(seed + 2), n_windows=n_windows)
    return {**base, "profile": profile, "null_poisson": null_poisson, "null_permutation": null_permutation}


def human_profile_rows(root: Path) -> list[dict]:
    """Every (dataset, structure, session), encoding and delay epochs,
    100 ms bins, W in {4, 6, 8}, each corpus's own native window -- both
    nulls in every row. 200 ms bins are not run here: coarser bins
    cost lags, the opposite of what resolving more window counts needs."""
    rows = []
    t0 = time.time()
    for i, meta in enumerate(iter_all_corpora(root)):
        for epoch in PROFILE_EPOCHS:
            window_s = meta["epoch_windows"][epoch]
            onset = meta["epoch_onsets"][epoch]
            counts = _counts_from_spikes(meta["spike_lists"], onset, window_s, PROFILE_BIN_MS)
            for n_windows in PROFILE_WINDOW_COUNTS:
                seed = _seed(meta["dataset"], meta["session"], meta["structure"], epoch, PROFILE_BIN_MS,
                             f"w{n_windows}", "native", "profile")
                run = _profile_run_row(counts, n_windows, seed)
                rows.append({
                    "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                    "structure": meta["structure"], "epoch": epoch, "bin_ms": PROFILE_BIN_MS,
                    "n_windows": n_windows, "window_mode": "native", "window_s": window_s, **run,
                })
        if (i + 1) % 10 == 0:
            print(f"  profile grid: session {i + 1}, {len(rows)} rows so far, "
                  f"{time.time() - t0:.1f}s elapsed", file=sys.stderr, flush=True)
    print(f"  profile grid done: {len(rows)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return rows


def alm_matched_geometry_rows(root: Path, human_window_s: float, rate_matched_keep_probability: float) -> list[dict]:
    """ALM re-run at the human window length (rather than its own native
    2.0 s) and the same W as the human grid, keeping the same rate-matched
    thinning (rate_matched_keep_probability) applied to the human grid so
    the level, not just the shape, stays comparable. Poisson null only --
    the permutation null answers a
    question about within-unit spike-train autocorrelation contaminating a
    shortest-lag existence claim that is only ambiguous in the human/
    macaque arms; ALM's own profile already decays across every lag with no
    shortest-lag ambiguity to resolve."""
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=PROFILE_BIN_MS, window_s=human_window_s, require_both_arms=False)
        if session is None:
            continue
        thin_rng = np.random.default_rng(_seed("alm_matched_geometry_thin", path.stem))
        counts = binomial_thin(session["control_counts"], rate_matched_keep_probability, thin_rng)
        for n_windows in PROFILE_WINDOW_COUNTS:
            seed = _seed("alm_matched_geometry", path.stem, n_windows)
            bins_per_window = counts.shape[2] // n_windows
            base = {"dataset": "inagaki_alm5", "patient": path.stem, "session": path.stem, "structure": "pooled",
                    "epoch": "delay", "bin_ms": PROFILE_BIN_MS, "n_windows": n_windows,
                    "window_mode": "human_matched", "window_s": human_window_s,
                    "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
                    "n_bins": int(counts.shape[2]), "bins_per_window": int(bins_per_window)}
            if counts.shape[0] < 8 or bins_per_window < PROFILE_MIN_BINS_PER_WINDOW:
                rows.append({**base, "profile": {"status": "bins_per_window_below_floor"}, "null_poisson": None})
                continue
            profile = r_gap_profile(counts, n_splits=PROFILE_N_SPLITS, rng=np.random.default_rng(seed), n_windows=n_windows)
            if profile["status"] != "fitted":
                rows.append({**base, "profile": profile, "null_poisson": None})
                continue
            null_poisson = poisson_null_r_gap_profile(
                counts, n_replicates=PROFILE_N_NULL_REPLICATES, n_splits_per_replicate=PROFILE_NULL_SPLITS_PER_REPLICATE,
                rng=np.random.default_rng(seed + 1), n_windows=n_windows)
            rows.append({**base, "profile": profile, "null_poisson": null_poisson})
    return rows


def _fitted_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["profile"].get("status") == "fitted"]


def _profile_lists(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Extract the matched (profile, poisson_null, permutation_null) gaps
    dicts from a list of fitted rows, keeping only rows where all three are
    present -- what :func:`state_persistence.classify_gap_profile` needs."""
    profiles, poisson_nulls, permutation_nulls = [], [], []
    for row in rows:
        if row["profile"].get("status") != "fitted" or row["null_poisson"] is None or row["null_permutation"] is None:
            continue
        profiles.append(row["profile"]["gaps"])
        poisson_nulls.append(row["null_poisson"]["gaps"])
        permutation_nulls.append(row["null_permutation"]["gaps"])
    return profiles, poisson_nulls, permutation_nulls


def double_difference_guard_gap1_gap3(rows_w4: list[dict]) -> dict:
    """A permanent guard on the r(1) - r(3) gap-effect contrast this claim's
    verdict turns on: the Poisson null's own r(1) - r(3) is not zero by
    construction (the estimator averages more window pairs at short gaps
    than long ones), so that contrast is tested against the null's own
    value of the same difference, not against zero."""
    profiles, poisson_nulls, _ = _profile_lists(rows_w4)
    obs, nul = [], []
    for g, n in zip(profiles, poisson_nulls):
        if 1 in g and 3 in g and 1 in n and 3 in n:
            obs.append(g[1]["r_median"] - g[3]["r_median"])
            nul.append(n[1]["r_null_median"] - n[3]["r_null_median"])
    return paired_vs_null_contrast(obs, nul, direction="greater")


def profile_decision(rows_by_w: dict[int, list[dict]]) -> dict:
    """Classify the pooled human delay/100ms profile at W=6 and W=8
    independently, then combine per the branch definitions --
    'graded_decay' fires if contrast B is significant and positive at
    EITHER W, since the branch is about whether a decay beyond the shortest
    lag is detectable once the lags are fine enough to see it, not about
    agreement between two window counts that measure different lag grids."""
    per_w = {}
    for w in (6, 8):
        rows = rows_by_w.get(w, [])
        profiles, poisson_nulls, permutation_nulls = _profile_lists(rows)
        per_w[str(w)] = classify_gap_profile(profiles, poisson_nulls, permutation_nulls, w_max=w)

    rotation_detected = any(per_w[k]["rotation_check"]["detected"] for k in per_w)
    any_shortest_lag_nuisance = any(
        per_w[k]["branch"] == "shortest_lag_is_nuisance" for k in per_w if per_w[k].get("contrast_c_shortest_lag_step_vs_permutation_null", {}).get("status") == "tested"
    )
    graded_decay_at_either_w = any(
        per_w[k]["contrast_b_decay_beyond_shortest_lag"].get("significant", False)
        and per_w[k]["contrast_b_decay_beyond_shortest_lag"].get("mean_diff", 0.0) > 0.0
        for k in per_w
    )
    c_tested = [per_w[k]["contrast_c_shortest_lag_step_vs_permutation_null"] for k in per_w
                if per_w[k]["contrast_c_shortest_lag_step_vs_permutation_null"].get("status") == "tested"]
    c_significant_everywhere_tested = bool(c_tested) and all(c.get("significant", False) for c in c_tested)

    if rotation_detected:
        overall_branch = "rotation"
    elif any_shortest_lag_nuisance:
        overall_branch = "shortest_lag_is_nuisance"
    elif graded_decay_at_either_w:
        overall_branch = "graded_decay"
    elif c_significant_everywhere_tested:
        overall_branch = "fast_component_plus_stable_floor"
    else:
        overall_branch = "unattributed_off_branch_list"

    return {
        "per_window_count": per_w,
        "overall_branch": overall_branch,
        "overall_branch_note": "rotation checked first; then whether ANY tested window count returns "
                                "'shortest_lag_is_nuisance', reported first regardless of the others because "
                                "it is the most consequential outcome available; then whether contrast B is "
                                "significant and positive at W=6 OR W=8 ('graded_decay'); then whether "
                                "contrast C is significant everywhere it was tested "
                                "('fast_component_plus_stable_floor'); anything else is reported off the "
                                "branch list explicitly rather than forced onto the nearest one.",
    }


def per_structure_profile(rows: list[dict], w_max: int, white_lookup: dict) -> dict:
    """The per-structure recut on the nuisance-immune terms only -- contrast
    A at every k >= 2 and contrast B -- plus the confound columns every
    per-structure observable here carries: Spearman correlation of
    the structure's per-session gap-effect proxy, r(2) - r(k_max), with
    unit count and with the session's nugget/white-noise fraction."""
    rows_w = [r for r in rows if r["n_windows"] == w_max]
    structures = sorted({r["structure"] for r in rows_w})
    k_max = w_max - 1
    out = {}
    for structure in structures:
        rows_s = [r for r in rows_w if r["structure"] == structure]
        patients = {r["patient"] for r in rows_s}
        profiles, poisson_nulls, permutation_nulls = _profile_lists(rows_s)
        classification = classify_gap_profile(profiles, poisson_nulls, permutation_nulls, w_max=w_max)

        gap_effect_proxy, unit_counts, whites = [], [], []
        for r in rows_s:
            if r["profile"].get("status") != "fitted" or k_max not in r["profile"]["gaps"] or 2 not in r["profile"]["gaps"]:
                continue
            gap_effect_proxy.append(r["profile"]["gaps"][2]["r_median"] - r["profile"]["gaps"][k_max]["r_median"])
            unit_counts.append(r["n_units"])
            key = (r["dataset"], r["session"], r["structure"], r["epoch"], r["bin_ms"])
            whites.append(white_lookup.get(key))

        confound = {"n_sessions_with_gap_effect": len(gap_effect_proxy)}
        if len(gap_effect_proxy) >= 6:
            corr_unit = spearman_permutation_test(np.array(gap_effect_proxy), np.array(unit_counts, dtype=float), rng=np.random.default_rng(0))
            confound["spearman_gap_effect_vs_unit_count"] = {"rho": corr_unit["rho"], "p_value": corr_unit["p_value"]}
            has_white = np.array([w is not None for w in whites])
            if has_white.sum() >= 6:
                white_arr = np.array([w if w is not None else np.nan for w in whites])
                corr_white = spearman_permutation_test(
                    np.array(gap_effect_proxy)[has_white], white_arr[has_white], rng=np.random.default_rng(1))
                confound["spearman_gap_effect_vs_nugget_fraction"] = {
                    "rho": corr_white["rho"], "p_value": corr_white["p_value"], "n": int(has_white.sum())}
            else:
                confound["spearman_gap_effect_vs_nugget_fraction"] = {"status": "not_enough_matched_variance_partition_rows"}
        else:
            confound["status"] = "not_enough_sessions_for_confound_check"
        confound["matched_count_subsampled_value"] = {
            "status": "not_computed",
            "reason": "requires re-fitting r(gap) per session at a common subsampled unit count; deferred "
                      "for wall-clock reasons -- the unit-count Spearman correlation above is "
                      "reported in its place and any structure whose |rho| exceeds its own between-structure "
                      "effect size should be read with that caveat, not promoted as clean.",
        }

        out[structure] = {
            "n_sessions": len(rows_s), "n_patients": len(patients),
            "existence_by_gap": classification["contrast_a_existence_by_gap"],
            "gap_effect": classification["contrast_b_decay_beyond_shortest_lag"],
            "confound": confound,
        }
    return out


def _profile_poisson_lists(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Like :func:`_profile_lists` but for rows that only ever carry a
    Poisson null (the ALM matched-geometry arm), not the per-unit
    permutation null."""
    profiles, poisson_nulls = [], []
    for row in rows:
        if row["profile"].get("status") != "fitted" or row.get("null_poisson") is None:
            continue
        profiles.append(row["profile"]["gaps"])
        poisson_nulls.append(row["null_poisson"]["gaps"])
    return profiles, poisson_nulls


def alm_matched_geometry_summary(rows: list[dict]) -> dict:
    by_w = {}
    for w in PROFILE_WINDOW_COUNTS:
        rows_w = [r for r in rows if r["n_windows"] == w]
        p, n = _profile_poisson_lists(rows_w)
        if not p:
            by_w[str(w)] = {"status": "not_available"}
            continue
        k_max = w - 1
        obs_b = [g[2]["r_median"] - g[k_max]["r_median"] for g in p if 2 in g and k_max in g]
        null_b = [nn[2]["r_null_median"] - nn[k_max]["r_null_median"] for g, nn in zip(p, n) if 2 in g and k_max in g and 2 in nn and k_max in nn]
        by_w[str(w)] = {
            "n_sessions": len(p),
            "median_profile": {gap: float(np.median([g[gap]["r_median"] for g in p if gap in g])) for gap in range(1, w)},
            "decay_beyond_shortest_lag": paired_vs_null_contrast(obs_b, null_b, direction="greater"),
        }
    return {"rows_n": len(rows), "by_window_count": by_w}


def heterogeneity_on_contrast_b(rows: list[dict], white_lookup: dict, w_max: int) -> dict:
    """The same three heterogeneity checks already applied to r(1) - r(3),
    re-pointed onto contrast B (r(2) - r(k_max)), the quantity the claim now
    rests on."""
    rows_w = [r for r in rows if r["n_windows"] == w_max]
    k_max = w_max - 1
    gap_effect, trial_count, unit_count, white_share = [], [], [], []
    for row in rows_w:
        if row["profile"].get("status") != "fitted" or 2 not in row["profile"]["gaps"] or k_max not in row["profile"]["gaps"]:
            continue
        gap_effect.append(row["profile"]["gaps"][2]["r_median"] - row["profile"]["gaps"][k_max]["r_median"])
        trial_count.append(row["n_trials"])
        unit_count.append(row["n_units"])
        key = (row["dataset"], row["session"], row["structure"], row["epoch"], row["bin_ms"])
        white_share.append(white_lookup.get(key))

    gap_effect = np.array(gap_effect)
    trial_count = np.array(trial_count, dtype=float)
    unit_count = np.array(unit_count, dtype=float)
    has_white = np.array([w is not None for w in white_share])
    white_share_arr = np.array([w if w is not None else np.nan for w in white_share])

    corr_trial = spearman_permutation_test(gap_effect, trial_count, rng=np.random.default_rng(0))
    corr_unit = spearman_permutation_test(gap_effect, unit_count, rng=np.random.default_rng(1))
    result = {
        "n_sessions": len(gap_effect), "w_max": w_max,
        "spearman_gap_effect_vs_trial_count": {"rho": corr_trial["rho"], "p_value": corr_trial["p_value"]},
        "spearman_gap_effect_vs_unit_count": {"rho": corr_unit["rho"], "p_value": corr_unit["p_value"]},
    }
    if has_white.sum() >= 4:
        corr_white = spearman_permutation_test(gap_effect[has_white], white_share_arr[has_white], rng=np.random.default_rng(2))
        result["spearman_gap_effect_vs_white_share"] = {"rho": corr_white["rho"], "p_value": corr_white["p_value"], "n": int(has_white.sum())}
    else:
        result["spearman_gap_effect_vs_white_share"] = {"status": "not_enough_matched_variance_partition_rows"}
    explained_by_trial_count = bool(corr_trial["p_value"] <= 0.05 and abs(corr_trial["rho"]) >= 0.5)
    result["explained_by_trial_count"] = explained_by_trial_count
    return result


def encoding_vs_delay_profile_discriminator(rows: list[dict], w_max: int) -> dict:
    """The within-session encoding-versus-delay attribution discriminator,
    re-run on contrasts B and C rather than the endpoint ratio r(1) - r(3):
    if C clears the permutation null in the human arms while B stays null,
    the honest reading is that the floor is general across epochs and the
    fast step is what differs between them -- a sentence outside the
    predeclared attribution branches computed elsewhere in this project,
    reported explicitly rather than forced onto the nearest one."""
    by_epoch = {}
    for epoch in PROFILE_EPOCHS:
        rows_e = [r for r in rows if r["epoch"] == epoch and r["structure"] == "pooled"
                  and r["n_windows"] == w_max and r["dataset"] in HUMAN_DATASETS]
        profiles, poisson_nulls, permutation_nulls = _profile_lists(rows_e)
        by_epoch[epoch] = classify_gap_profile(profiles, poisson_nulls, permutation_nulls, w_max=w_max)
    return by_epoch


def _gap_value(row: dict, gap: int) -> float | None:
    profile = row["profile"]
    if profile["status"] != "fitted" or gap not in profile["gaps"]:
        return None
    return profile["gaps"][gap]["r_median"]


def _null_gap_value(row: dict, gap: int, field: str) -> float | None:
    if row["null"] is None or gap not in row["null"]["gaps"]:
        return None
    return row["null"]["gaps"][gap][field]


def _existence_contrast(rows: list[dict], gap: int, direction: str = "greater") -> dict:
    observed, null = [], []
    for row in rows:
        obs = _gap_value(row, gap)
        nul = _null_gap_value(row, gap, "r_null_median")
        if obs is None or nul is None:
            continue
        observed.append(obs)
        null.append(nul)
    n_pairs = len(observed)
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p}
    test = paired_sign_flip_test(np.array(observed), np.array(null), alternative=direction)
    significant = bool(test["p_value"] <= 0.05)

    exceed_count = 0
    exceed_n = 0
    for row in rows:
        obs = _gap_value(row, gap)
        p95 = _null_gap_value(row, gap, "r_null_p95")
        if obs is None or p95 is None:
            continue
        exceed_n += 1
        if (obs > p95) if direction == "greater" else (obs < p95):
            exceed_count += 1
    binom = stats.binomtest(exceed_count, exceed_n, 0.05, alternative="greater") if exceed_n > 0 else None

    return {
        "status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"], "significant": significant,
        "exceedance_count": exceed_count, "exceedance_n": exceed_n,
        "exceedance_p_value": float(binom.pvalue) if binom is not None else None,
        "exceedance_significant": bool(binom is not None and binom.pvalue <= 0.05),
    }


def _gap_effect_contrast(rows: list[dict]) -> dict:
    r1, r3 = [], []
    for row in rows:
        a, b = _gap_value(row, 1), _gap_value(row, 3)
        if a is None or b is None:
            continue
        r1.append(a)
        r3.append(b)
    n_pairs = len(r1)
    min_attainable_p = 1.0 / (2 ** n_pairs) if n_pairs > 0 else 1.0
    if n_pairs < 4 or min_attainable_p > 0.05:
        return {"status": "underpowered_by_construction", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p}
    test = paired_sign_flip_test(np.array(r1), np.array(r3), alternative="greater")
    return {
        "status": "tested", "n_pairs": n_pairs, "min_attainable_p": min_attainable_p,
        "mean_diff": test["mean_diff"], "p_value": test["p_value"],
        "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "significant": bool(test["p_value"] <= 0.05),
        "n_r1_gt_r3": int(sum(a > b for a, b in zip(r1, r3))),
    }


def deciding_contrast(delay_pooled_rows: list[dict], matched_sensitivity: dict) -> dict:
    contrast_a_gap1 = _existence_contrast(delay_pooled_rows, 1, direction="greater")
    contrast_a_gap3_below_null = _existence_contrast(delay_pooled_rows, 3, direction="less")
    contrast_b = _gap_effect_contrast(delay_pooled_rows)

    a_significant = contrast_a_gap1.get("significant", False)
    rotates = (
        contrast_a_gap3_below_null.get("significant", False) and a_significant
    )

    rate_matched_key = None
    if matched_sensitivity.get("status") == "tested":
        rate_matched_key = str(round(matched_sensitivity["rate_matched_keep_probability"], 4))
    power = None
    if rate_matched_key is not None:
        regime = matched_sensitivity["regimes"].get(rate_matched_key)
        if regime and regime.get("status") == "tested":
            power = regime["power_at_0p05_resampled_to_n18"]
    sensitivity_confirmed = power is not None and power >= 0.80

    if not a_significant:
        branch = "no_trial_specific_state"
    elif rotates:
        branch = "state_rotates"
    elif contrast_b.get("significant", False):
        branch = "state_drifts"
    elif sensitivity_confirmed:
        branch = "state_persists_timescale_unresolved"
    else:
        branch = "state_persists_timescale_unresolved_sensitivity_unconfirmed"

    return {
        "reachability": reachability_note(),
        "contrast_a_existence_gap1": contrast_a_gap1,
        "contrast_a_rotation_check_gap3_below_null": contrast_a_gap3_below_null,
        "contrast_b_gap_effect_r1_minus_r3": contrast_b,
        "matched_sensitivity_power_at_rate_matched_regime": power,
        "matched_sensitivity_confirmed_at_0p80": sensitivity_confirmed,
        "branch": branch,
        "branch_priority_note": "Evaluated in this order: existence at gap=1 first, since its absence is the "
                                 "most consequential outcome and is reported before anything else; then the "
                                 "rotation signature (r(3) significantly "
                                 "below its own null while r(1) is above); then the gap-effect contrast for a "
                                 "positive decay; then the matched-sensitivity confirmation that turns a "
                                 "non-significant gap effect into a bound rather than a non-result.",
    }


def heterogeneity(delay_pooled_rows: list[dict], white_lookup: dict) -> dict:
    gap_effect, trial_count, unit_count, white_share = [], [], [], []
    for row in delay_pooled_rows:
        r1, r3 = _gap_value(row, 1), _gap_value(row, 3)
        if r1 is None or r3 is None:
            continue
        key = (row["dataset"], row["session"], row["structure"], row["epoch"], row["bin_ms"])
        white = white_lookup.get(key)
        gap_effect.append(r1 - r3)
        trial_count.append(row["n_trials"])
        unit_count.append(row["n_units"])
        white_share.append(white)
    gap_effect = np.array(gap_effect)
    trial_count = np.array(trial_count, dtype=float)
    unit_count = np.array(unit_count, dtype=float)
    has_white = np.array([w is not None for w in white_share])
    white_share_arr = np.array([w if w is not None else np.nan for w in white_share])

    corr_trial = spearman_permutation_test(gap_effect, trial_count, rng=np.random.default_rng(0))
    corr_unit = spearman_permutation_test(gap_effect, unit_count, rng=np.random.default_rng(1))
    result = {
        "n_sessions": len(gap_effect),
        "spearman_gap_effect_vs_trial_count": {"rho": corr_trial["rho"], "p_value": corr_trial["p_value"]},
        "spearman_gap_effect_vs_unit_count": {"rho": corr_unit["rho"], "p_value": corr_unit["p_value"]},
    }
    if has_white.sum() >= 4:
        corr_white = spearman_permutation_test(gap_effect[has_white], white_share_arr[has_white], rng=np.random.default_rng(2))
        result["spearman_gap_effect_vs_white_share"] = {"rho": corr_white["rho"], "p_value": corr_white["p_value"], "n": int(has_white.sum())}
    else:
        result["spearman_gap_effect_vs_white_share"] = {"status": "not_enough_matched_variance_partition_rows"}

    explained_by_trial_count = bool(corr_trial["p_value"] <= 0.05 and abs(corr_trial["rho"]) >= 0.5)
    result["explained_by_trial_count"] = explained_by_trial_count
    result["branch_taken"] = (
        "(a) only: trial count explains the between-session heterogeneity, so the spread is treated as noise "
        "and (b)/(c) are not reported as an anatomical or load/accuracy claim." if explained_by_trial_count else
        "(a) does not explain the spread; (b) per-structure existence and gap effect are reported in "
        "per_structure_delay; (c) is not computable from the current corpus_sessions.py iteration surface, see "
        "load_accuracy_arm below."
    )
    result["load_accuracy_arm"] = {
        "status": "not_computable",
        "reason": "iter_dandi_000469 filters to loads==1 trials only and all three human iterators filter to "
                  "correct/artifact-free trials only, so neither load nor accuracy carries residual per-trial "
                  "variance in the population this module receives without changing corpus_sessions.py's "
                  "existing filtering, which this module does not do.",
    }
    return result


def encoding_vs_delay_discriminator(primary_rows: list[dict]) -> dict:
    by_epoch = {
        epoch: [r for r in primary_rows if r["epoch"] == epoch and r["structure"] == "pooled"
                and r["bin_ms"] == 100.0 and r["dataset"] in HUMAN_DATASETS]
        for epoch in ("encoding", "delay")
    }
    return {
        "encoding": {
            "existence_gap1": _existence_contrast(by_epoch["encoding"], 1),
            "gap_effect": _gap_effect_contrast(by_epoch["encoding"]),
            "n_sessions": len(by_epoch["encoding"]),
        },
        "delay": {
            "existence_gap1": _existence_contrast(by_epoch["delay"], 1),
            "gap_effect": _gap_effect_contrast(by_epoch["delay"]),
            "n_sessions": len(by_epoch["delay"]),
        },
    }


def attribution_verdict(encoding_vs_delay: dict, panichello: dict, matched_sensitivity_power: float | None) -> dict:
    encoding_resolves = encoding_vs_delay["encoding"]["gap_effect"].get("significant", False)
    delay_resolves = encoding_vs_delay["delay"]["gap_effect"].get("significant", False)

    panichello_rows_delay = [r for r in panichello.get("rows", []) if r["bin_ms"] == 100.0]
    panichello_gap_effect = _gap_effect_contrast(panichello_rows_delay) if panichello_rows_delay else \
        {"status": "not_available"}
    panichello_resolves = panichello_gap_effect.get("significant", False)

    panichello_tested = panichello_gap_effect.get("status") == "tested"
    if encoding_resolves and not delay_resolves:
        branch = "maintenance_specific"
    elif (not encoding_resolves) and (not delay_resolves) and panichello_resolves:
        branch = "human_regime_specific"
    elif (not encoding_resolves) and (not delay_resolves) and panichello_tested and not panichello_resolves:
        branch = "task_class_specific"
    elif encoding_resolves and delay_resolves and panichello_tested and panichello_resolves:
        # Off the four predeclared branches: none of them describes "every arm
        # resolves". Reported explicitly rather than forced onto "unattributed", which would
        # wrongly imply the finding is confined to human MTL/frontal verbal maintenance when it
        # generalises to an independent macaque prefrontal delay-period population as well.
        branch = "drift_resolves_in_every_arm_tested"
    else:
        branch = "unattributed"

    return {
        "encoding_gap_effect_resolves": encoding_resolves,
        "delay_gap_effect_resolves": delay_resolves,
        "panichello_gap_effect": panichello_gap_effect,
        "matched_sensitivity_power_reference": matched_sensitivity_power,
        "branch": branch,
        "scope_statement": "human MTL and frontal, verbal maintenance" if branch == "unattributed" else (
            "the drift resolves in human encoding, human delay, and macaque prefrontal delay alike, so "
            "species, region, and task are not distinguishing it -- no narrower scope statement is licensed "
            "or required" if branch == "drift_resolves_in_every_arm_tested" else None
        ),
    }


def per_structure_delay(primary_rows: list[dict]) -> dict:
    rows_100 = [r for r in primary_rows if r["epoch"] == "delay" and r["bin_ms"] == 100.0]
    structures = sorted({r["structure"] for r in rows_100})
    out = {}
    for structure in structures:
        rows_s = [r for r in rows_100 if r["structure"] == structure]
        patients = {r["patient"] for r in rows_s}
        out[structure] = {
            "n_sessions": len(rows_s), "n_patients": len(patients),
            "existence_gap1": _existence_contrast(rows_s, 1),
            "existence_gap2": _existence_contrast(rows_s, 2),
            "existence_gap3": _existence_contrast(rows_s, 3),
            "gap_effect": _gap_effect_contrast(rows_s),
        }
    return out


def _load_white_share_lookup() -> dict:
    if not VARIANCE_PARTITION_PATH.exists():
        return {}
    data = json.loads(VARIANCE_PARTITION_PATH.read_text())
    lookup = {}
    for row in data.get("human_rows", []):
        key = (row["dataset"], row["session"], row["structure"], row["epoch"], row["bin_ms"])
        lookup[key] = row["partition"].get("white_fraction_median")
    return lookup


def main() -> None:
    root = data_root()
    t_start = time.time()

    print("Human primary grid...", file=sys.stderr)
    primary_rows = human_primary_rows(root)

    print("Human delay secondary grid (floor window, W=3)...", file=sys.stderr)
    delay_secondary_rows = human_delay_secondary_rows(root)

    print("ALM comparison rows...", file=sys.stderr)
    alm_rows = alm_census_rows(root)
    print(f"  {len(alm_rows)} ALM rows, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    delay_pooled = [r for r in primary_rows if r["epoch"] == "delay" and r["bin_ms"] == 100.0
                    and r["structure"] == "pooled" and r["dataset"] in HUMAN_DATASETS]
    human_median_units = int(np.median([r["n_units"] for r in delay_pooled])) if delay_pooled else 0
    human_median_trials = int(np.median([r["n_trials"] for r in delay_pooled])) if delay_pooled else 0
    human_rates = []
    for meta in iter_all_corpora(root):
        if meta["structure"] != "pooled":
            continue
        counts = _counts_from_spikes(meta["spike_lists"], meta["epoch_onsets"]["delay"], meta["epoch_windows"]["delay"], 100.0)
        human_rates.append(firing_rate_hz(counts, meta["epoch_windows"]["delay"]))
    human_median_rate_hz = float(np.median(human_rates)) if human_rates else 0.0

    print(f"human delay/100ms/pooled: n={len(delay_pooled)} median_units={human_median_units} "
          f"median_trials={human_median_trials} median_rate_hz={human_median_rate_hz:.2f}", file=sys.stderr)

    print("Matched-sensitivity ALM arm...", file=sys.stderr)
    matched_sensitivity = matched_sensitivity_alm(root, human_median_units, human_median_trials, human_median_rate_hz)
    print(f"  done, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    print("Panichello macaque prefrontal arm...", file=sys.stderr)
    panichello = panichello_rows(root)
    print(f"  {len(panichello.get('rows', []))} Panichello rows, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    decision = deciding_contrast(delay_pooled, matched_sensitivity)
    het = heterogeneity(delay_pooled, _load_white_share_lookup())
    encoding_vs_delay = encoding_vs_delay_discriminator(primary_rows)
    rate_matched_power = decision["matched_sensitivity_power_at_rate_matched_regime"]
    attribution = attribution_verdict(encoding_vs_delay, panichello, rate_matched_power)
    per_structure = per_structure_delay(primary_rows)

    print("Calibration ladder (high-SNR and human-yield regimes)...", file=sys.stderr)
    calibration_high_snr = calibration_ladder(150, 45, 23, 0.1, rate_hz=1.0, seed=100)
    calibration_human_yield = calibration_ladder(40, 36, 23, 0.1, rate_hz=0.15, seed=101)

    output = {
        "version": "2026-08-11",
        "scope": (
            "Human corpora: DANDI 000469, 001187, 000574 via src/corpus_sessions.iter_all_corpora. Primary "
            "grid: all four epochs, both bin widths, each corpus's own native window, 4 sub-windows. Delay "
            "epoch additionally gets the common 2.3 s floor window and 3 sub-windows at the native window "
            "(both bin widths) -- a documented narrowing of the full cross-product (every epoch x both "
            "window modes x both window counts) to the epoch the deciding contrast and the "
            "anatomical comparison actually use, for wall-clock reasons. Mouse ALM (Inagaki) is the "
            "matched-sensitivity reference. Panichello 2024 macaque lPFC delay is the second attribution "
            "discriminator, pooled per session (no per-channel area label exists in the deposited data). "
            "LFP corpora and the Watters macaque arm are not run here, a deferral carried forward from "
            "earlier in this project unchanged."
        ),
        "n_splits": N_SPLITS, "n_null_replicates": N_NULL_REPLICATES,
        "matched_sensitivity_draws": MATCHED_POWER_DRAWS, "power_resamples": POWER_RESAMPLES,
        "calibration_ladder_high_snr": calibration_high_snr,
        "calibration_ladder_human_yield": calibration_human_yield,
        "deciding_contrast": decision,
        "heterogeneity": het,
        "per_structure_delay": per_structure,
        "attribution": {
            "encoding_vs_delay": encoding_vs_delay,
            "panichello_macaque_prefrontal": panichello,
            "macaque_maintenance_watters": {
                "status": "not_run",
                "reason": "Fallback only if the first two discriminators do not resolve; not reached here.",
            },
            "verdict": attribution,
        },
        "lfp_corpora": {
            "status": "not_run",
            "reason": "ds004752, ds005489, ds005557 and the tACS corpus are not yet wired into "
                      "src/corpus_sessions.py's iteration surface. Deferred, not dropped.",
        },
        "matched_sensitivity_alm": matched_sensitivity,
        "human_median_units_delay_100ms": human_median_units,
        "human_median_trials_delay_100ms": human_median_trials,
        "human_median_rate_hz_delay": human_median_rate_hz,
        "human_primary_rows": primary_rows,
        "human_delay_secondary_rows": delay_secondary_rows,
        "alm_rows": alm_rows,
        "wall_clock_s": time.time() - t_start,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(json.dumps(decision, indent=2))


def main_profile() -> None:
    """Does the shortest-lag step in results/state_persistence.json's gap
    effect reflect a real trial-specific state or within-unit spike-train
    autocorrelation that the Poisson null cannot see? Extends the r(gap)
    profile to more window counts, adds a second null sensitive to
    within-unit temporal autocorrelation, and applies a decision rule
    stated on the whole profile rather than an endpoint ratio. Writes
    results/state_persistence_profile.json beside (never in place of)
    results/state_persistence.json."""
    root = data_root()
    t_start = time.time()

    print("Permutation null validation on synthetic data...", file=sys.stderr)
    null_validation = permutation_null_validation()
    print(f"  usable={null_validation['usable']}", file=sys.stderr)
    if not null_validation["usable"]:
        output = {
            "version": "2026-08-12",
            "status": "stopped_after_null_validation_failure",
            "permutation_null_validation": null_validation,
            "note": "The per-unit permutation null failed its own pre-registered validation on synthetic "
                    "data and cannot be used to clear or convict the shortest-lag step. This is a "
                    "completed deliverable, not a failure -- see the check that failed above.",
            "wall_clock_s": time.time() - t_start,
        }
        PROFILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PROFILE_OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {PROFILE_OUTPUT_PATH} (validation failure) in {time.time() - t_start:.1f}s", file=sys.stderr)
        return

    print("Human profile grid (W=4,6,8, both nulls, encoding+delay, 100ms)...", file=sys.stderr)
    primary_rows = human_profile_rows(root)

    prior_path = OUTPUT_PATH  # results/state_persistence.json, a read-only input here
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    rate_matched_keep_probability = prior.get("matched_sensitivity_alm", {}).get("rate_matched_keep_probability", 1.0)
    human_window_s = EPOCH_WINDOWS_S["delay"]

    print("ALM matched-geometry arm...", file=sys.stderr)
    alm_rows = alm_matched_geometry_rows(root, human_window_s, rate_matched_keep_probability)
    print(f"  {len(alm_rows)} ALM matched-geometry rows, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    delay_pooled_all_w = [r for r in primary_rows if r["epoch"] == "delay" and r["structure"] == "pooled"
                           and r["dataset"] in HUMAN_DATASETS]
    delay_pooled_by_w = {w: [r for r in delay_pooled_all_w if r["n_windows"] == w] for w in PROFILE_WINDOW_COUNTS}

    print("Predeclared profile decision rule (W=6, W=8)...", file=sys.stderr)
    decision = profile_decision(delay_pooled_by_w)
    double_difference_guard = double_difference_guard_gap1_gap3(delay_pooled_by_w[4])

    white_lookup = _load_white_share_lookup()
    print("Per-structure recut (nuisance-immune terms)...", file=sys.stderr)
    delay_rows = [r for r in primary_rows if r["epoch"] == "delay"]
    per_structure_w4 = per_structure_profile(delay_rows, 4, white_lookup)
    per_structure_w8 = per_structure_profile(delay_rows, 8, white_lookup)

    print("Heterogeneity on contrast B...", file=sys.stderr)
    heterogeneity_w8 = heterogeneity_on_contrast_b(delay_pooled_all_w, white_lookup, 8)

    print("ALM matched-geometry summary...", file=sys.stderr)
    alm_summary = alm_matched_geometry_summary(alm_rows)

    print("Encoding-vs-delay discriminator on contrasts B/C (opportunistic; a full attribution verdict "
          "across species, region and task is not run here)...", file=sys.stderr)
    encoding_vs_delay = encoding_vs_delay_profile_discriminator(primary_rows, 8)

    output = {
        "version": "2026-08-12",
        "supersedes_endpoint_ratio_reading_in": "results/state_persistence.json's deciding_contrast "
                                                 "(branch 'state_drifts'), per this file's own decision below. "
                                                 "results/state_persistence.json is not recomputed or edited.",
        "scope": (
            "Every (dataset, structure, session) reachable through src/corpus_sessions.iter_all_corpora; "
            "encoding and delay epochs; 100 ms bins only (200 ms is not run here -- it costs lags); "
            "W in {4, 6, 8} at each corpus's own native window. ALM (Inagaki) re-run at the human delay "
            "floor window (2.3 s) with the same rate-matched thinning applied throughout this grid, same "
            "W grid, Poisson null only. Panichello and a full species/region/task attribution verdict are "
            "not run by this analysis, which is scoped to the shortest-lag question alone; the "
            "encoding-vs-delay field below is an opportunistic by-product of the primary grid already "
            "covering both epochs, not a completed attribution deliverable."
        ),
        "n_splits": PROFILE_N_SPLITS, "n_null_replicates": PROFILE_N_NULL_REPLICATES,
        "window_counts": list(PROFILE_WINDOW_COUNTS),
        "permutation_null_validation": null_validation,
        "double_difference_guard_gap1_gap3_w4": double_difference_guard,
        "deciding_contrast": decision,
        "per_structure_delay": {"w4": per_structure_w4, "w8": per_structure_w8},
        "heterogeneity_on_contrast_b": heterogeneity_w8,
        "alm_matched_geometry": {"human_window_s": human_window_s,
                                  "rate_matched_keep_probability": rate_matched_keep_probability,
                                  "summary": alm_summary},
        "encoding_vs_delay_on_contrasts_b_c_w8": encoding_vs_delay,
        "human_profile_rows": primary_rows,
        "alm_matched_geometry_rows": alm_rows,
        "wall_clock_s": time.time() - t_start,
    }

    PROFILE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROFILE_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {PROFILE_OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(json.dumps({"branch": decision["overall_branch"]}, indent=2))


LAG_WIDTHS_BINS = (2, 3, 5)
LAG_BIN_MS = 100.0
LAG_EPOCHS = ("encoding", "delay")
LAG_N_SPLITS = 12
LAG_N_NULL_REPLICATES = 20
LAG_NULL_SPLITS_PER_REPLICATE = 6
LAG_MATCHED_COUNT_N_SPLITS = 8
LAG_MATCHED_COUNT_N_NULL_REPLICATES = 10
LAG_DECIDING_WIDTH_BINS = 3
LAG_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_lag.json"
# The macaque prefrontal lag arm's split and permutation-replicate counts, named explicitly for the same
# large-simultaneous-population wall-clock reason as PANICHELLO_PROFILE_N_SPLITS above. This arm carries
# this project's cross-species d_perm-slope comparison, so its settings differing from LAG_N_SPLITS/
# LAG_N_NULL_REPLICATES is a disclosed, measured mismatch (see
# scripts/run_persistence_estimator_split_count_sensitivity.py) rather than an unexamined economy.
PANICHELLO_LAG_N_SPLITS = 10
PANICHELLO_LAG_N_NULL_REPLICATES = 10


def _lag_run_row(counts: np.ndarray, width_bins: int, seed: int,
                  n_splits: int = LAG_N_SPLITS, n_null_replicates: int = LAG_N_NULL_REPLICATES) -> dict:
    """A session/width row at fixed window width, both nulls -- never a
    silent omission: a session that cannot support this width gets a status,
    not a missing row. ``width_exceeds_epoch`` is the fixed-width analogue
    of the window-count profile's bins-per-window floor -- the shortest
    reachable lag is the width itself, so at least 2 * width bins are
    required for even one lag to exist."""
    n_trials, n_units, n_bins = counts.shape
    base = {"n_trials": int(n_trials), "n_units": int(n_units), "n_bins": int(n_bins), "width_bins": int(width_bins)}
    if n_trials < 8:
        return {**base, "profile": {"status": "fewer_than_eight_trials"}, "null_poisson": None, "null_permutation": None}
    if n_bins < 2 * width_bins:
        return {**base, "profile": {"status": "width_exceeds_epoch"}, "null_poisson": None, "null_permutation": None}
    profile = r_lag_profile(counts, width_bins, n_splits=n_splits, rng=np.random.default_rng(seed))
    if profile["status"] != "fitted":
        return {**base, "profile": profile, "null_poisson": None, "null_permutation": None}
    null_poisson = poisson_null_r_lag_profile(
        counts, width_bins, n_replicates=n_null_replicates, n_splits_per_replicate=LAG_NULL_SPLITS_PER_REPLICATE,
        rng=np.random.default_rng(seed + 1))
    null_permutation = per_unit_permutation_null_r_lag_profile(
        counts, width_bins, n_replicates=n_null_replicates, n_splits_per_replicate=LAG_NULL_SPLITS_PER_REPLICATE,
        rng=np.random.default_rng(seed + 2))
    return {**base, "profile": profile, "null_poisson": null_poisson, "null_permutation": null_permutation}


def human_lag_rows(root: Path) -> tuple[list[dict], dict]:
    """Every (dataset, structure, session) reachable through
    corpus_sessions.iter_all_corpora, encoding and delay epochs, 100 ms
    bins, width in {2, 3, 5} bins -- both nulls in every row. Also caches
    each session's own delay-epoch counts array, keyed by (dataset,
    session, structure), for the matched-unit-count per-structure
    subsample, which needs the raw counts rather than an already-fitted
    profile."""
    rows = []
    delay_counts_cache: dict[tuple, np.ndarray] = {}
    t0 = time.time()
    for i, meta in enumerate(iter_all_corpora(root)):
        for epoch in LAG_EPOCHS:
            window_s = meta["epoch_windows"][epoch]
            onset = meta["epoch_onsets"][epoch]
            counts = _counts_from_spikes(meta["spike_lists"], onset, window_s, LAG_BIN_MS)
            if epoch == "delay":
                delay_counts_cache[(meta["dataset"], meta["session"], meta["structure"])] = counts
            for width in LAG_WIDTHS_BINS:
                seed = _seed(meta["dataset"], meta["session"], meta["structure"], epoch, LAG_BIN_MS,
                             f"width{width}", "lag")
                run = _lag_run_row(counts, width, seed)
                rows.append({
                    "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
                    "structure": meta["structure"], "epoch": epoch, "bin_ms": LAG_BIN_MS,
                    "window_mode": "native", "window_s": window_s, **run,
                })
        if (i + 1) % 10 == 0:
            print(f"  lag grid: session {i + 1}, {len(rows)} rows so far, "
                  f"{time.time() - t0:.1f}s elapsed", file=sys.stderr, flush=True)
    print(f"  lag grid done: {len(rows)} rows in {time.time() - t0:.1f}s", file=sys.stderr)
    return rows, delay_counts_cache


def alm_lag_rows(root: Path, human_window_s: float, rate_matched_keep_probability: float) -> list[dict]:
    """ALM re-run at the human delay window and the fixed-width estimator,
    keeping the same rate-matched thinning applied throughout this grid --
    WITH BOTH nulls, unlike the window-count profile arm's ALM run, closing
    exactly the gap that arm left open."""
    directory = alm_data_directory(root)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("*.mat")):
        session = load_alm_raw_session(path, bin_ms=LAG_BIN_MS, window_s=human_window_s, require_both_arms=False)
        if session is None:
            continue
        thin_rng = np.random.default_rng(_seed("alm_lag_thin", path.stem))
        counts = binomial_thin(session["control_counts"], rate_matched_keep_probability, thin_rng)
        for width in LAG_WIDTHS_BINS:
            seed = _seed("alm_lag", path.stem, width)
            run = _lag_run_row(counts, width, seed)
            rows.append({
                "dataset": "inagaki_alm5", "patient": path.stem, "session": path.stem, "structure": "pooled",
                "epoch": "delay", "bin_ms": LAG_BIN_MS, "window_mode": "human_matched", "window_s": human_window_s,
                **run,
            })
    return rows


def panichello_lag_rows(root: Path, n_splits: int = PANICHELLO_LAG_N_SPLITS,
                         n_null_replicates: int = PANICHELLO_LAG_N_NULL_REPLICATES) -> dict:
    """Macaque lPFC delay, the same fixed-width estimator, WITH BOTH nulls --
    what any cross-species sentence about a decay (as opposed to mere
    existence) needs before it is licensed in either direction.
    ``n_splits``/``n_null_replicates`` default to this arm's lighter
    settings (PANICHELLO_LAG_N_SPLITS above) for the same large-simultaneous-
    population wall-clock reason as the window-count profile arm, but are
    explicit parameters: this arm carries this project's cross-species
    d_perm-slope comparison, so it must be callable at the human/ALM module
    default (LAG_N_SPLITS, LAG_N_NULL_REPLICATES) when a comparison needs
    matched settings rather than a documented, disclosed mismatch -- see
    scripts/run_persistence_estimator_split_count_sensitivity.py, which
    calls this arm at both regimes."""
    directory = _panichello_directory(root)
    if directory is None:
        return {"status": "not_available", "reason": "Panichello_2024 not staged", "rows": []}
    files = sorted(glob.glob(str(directory / "*.mat")))
    if not files:
        return {"status": "not_available", "reason": "no .mat files found", "rows": []}

    rows = []
    for path in files:
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
        for width in LAG_WIDTHS_BINS:
            seed = _seed("panichello_lag", Path(path).stem, width)
            run = _lag_run_row(counts, width, seed, n_splits=n_splits, n_null_replicates=n_null_replicates)
            rows.append({
                "dataset": "panichello_2024", "patient": Path(path).stem, "session": Path(path).stem,
                "structure": "pooled", "epoch": "delay", "bin_ms": LAG_BIN_MS, "window_mode": "native",
                "window_s": (PANICHELLO_DELAY_WINDOW_MS[1] - PANICHELLO_DELAY_WINDOW_MS[0]) / 1000.0, **run,
            })
    return {
        "status": "tested", "rows": rows,
        "estimator_settings": {
            "n_splits": n_splits, "n_null_replicates": n_null_replicates,
            "null_splits_per_replicate": LAG_NULL_SPLITS_PER_REPLICATE,
            "differs_from_human_and_alm_module_default": (n_splits, n_null_replicates) != (LAG_N_SPLITS, LAG_N_NULL_REPLICATES),
            "human_and_alm_module_default": {"n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES},
        },
    }


def _lag_lists(rows: list[dict], width: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Matched (profile.lags, null_poisson.lags, null_permutation.lags) for
    fitted rows at a given width -- what classify_lag_profile needs."""
    profiles, pois, perm = [], [], []
    for r in rows:
        if r.get("width_bins") != width:
            continue
        if r["profile"].get("status") != "fitted" or r["null_poisson"] is None or r["null_permutation"] is None:
            continue
        profiles.append(r["profile"]["lags"])
        pois.append(r["null_poisson"]["lags"])
        perm.append(r["null_permutation"]["lags"])
    return profiles, pois, perm


def lag_decision(delay_pooled_rows: list[dict]) -> dict:
    """Classifies the pooled human delay/100ms profile at every window count
    this grid runs, with w=3 nominated as the deciding arm -- the width that
    reaches the most lags at above-floor bins per window -- and the other
    widths reported alongside it as a cross-width check rather than
    silently reconciled."""
    per_width = {}
    for width in LAG_WIDTHS_BINS:
        profiles, pois, perm = _lag_lists(delay_pooled_rows, width)
        per_width[str(width)] = classify_lag_profile(profiles, pois, perm, width_bins=width, bin_width_s=LAG_BIN_MS / 1000.0)
    branch_by_width = {w: per_width[w]["branch"] for w in per_width}
    widths_agree = len(set(branch_by_width.values())) == 1
    return {
        "per_width": per_width,
        "deciding_width_bins": LAG_DECIDING_WIDTH_BINS,
        "deciding_branch": per_width[str(LAG_DECIDING_WIDTH_BINS)]["branch"],
        "branch_by_width": branch_by_width,
        "widths_agree": widths_agree,
        "most_bins_per_window_width_bins": max(LAG_WIDTHS_BINS),
        "reconciliation_note": (
            "w=3 bins (300 ms) is named the deciding arm on reachability grounds -- it is the width that "
            "reaches the most lags at above-floor bins per window -- not on the precision grounds a wider "
            "window would give (a wider window's less-attenuated correlation is a separate, later "
            "consideration; see results/state_shape_common_range.json's width reconciliation for that "
            "comparison). w=2 and w=5 are reported alongside it as a cross-width check; if they disagree with "
            "the w=3 verdict that disagreement is reported in branch_by_width rather than silently resolved."
        ),
    }


def patient_clustered_human_delay(delay_pooled_rows: list[dict]) -> dict:
    """The live inferential F1 companion to the retained session profiles.

    The raw estimator is necessarily evaluated session by session, but a
    participant with several recording sessions does not supply several
    independent human replications.  This wrapper makes that distinction
    explicit and keeps the historical session-row summaries available only
    as descriptive profiles.
    """
    by_width = {
        str(width): patient_clustered_lag_inference(
            delay_pooled_rows, width, LAG_BIN_MS / 1000.0, seed=_seed("patient_clustered_lag", width))
        for width in LAG_WIDTHS_BINS
    }
    deciding = by_width[str(LAG_DECIDING_WIDTH_BINS)]
    return {
        "status": "tested",
        "scope": (
            "Human delay, pooled structure only, DANDI 000469/000574/001187. The stored session profiles "
            "are retained as descriptive estimator output; the fields below are the current confirmatory "
            "inference because they use patients, not sessions, as independent units."
        ),
        "by_width": by_width,
        "deciding_width_bins": LAG_DECIDING_WIDTH_BINS,
        "deciding_branch": deciding["branch"],
        "n_patients_at_deciding_width": deciding["n_patients"],
        "n_session_rows_descriptive_at_deciding_width": deciding["n_session_rows_descriptive"],
        "branch_by_width": {width: result["branch"] for width, result in by_width.items()},
    }


def main_patient_clustered_lag() -> None:
    """Refresh only the patient-clustered F1 field from stored raw profiles.

    This is a proper re-analysis of the saved session profiles, not a hand
    edit: no raw-data census is needed because every observed and matched
    null lag value used by the cluster analysis is already provenance-bound
    in ``human_lag_rows``.  A lock prevents a concurrent result writer from
    losing this narrow update.
    """
    with locked_json_update(LAG_OUTPUT_PATH) as artifact:
        rows = artifact.get("human_lag_rows", [])
        delay_pooled = [r for r in rows if r.get("epoch") == "delay" and r.get("structure") == "pooled"
                        and r.get("dataset") in HUMAN_DATASETS]
        if not delay_pooled:
            raise RuntimeError("state_persistence_lag.json has no fitted human pooled delay rows")
        artifact["patient_clustered_human_delay"] = patient_clustered_human_delay(delay_pooled)
        artifact["patient_clustered_inference_note"] = (
            "Added 2026-09-01. Existing session-row fields remain descriptive because 72 delay rows nest "
            "within 52 dataset-qualified patients. Current human F1 inference is patient_clustered_human_delay."
        )
    result = json.loads(LAG_OUTPUT_PATH.read_text())["patient_clustered_human_delay"]
    print(json.dumps({
        "deciding_width_bins": result["deciding_width_bins"],
        "n_patients": result["n_patients_at_deciding_width"],
        "n_session_rows_descriptive": result["n_session_rows_descriptive_at_deciding_width"],
        "branch": result["deciding_branch"],
        "branch_by_width": result["branch_by_width"],
    }, indent=2))


def _session_key(row: dict) -> tuple:
    return (row["dataset"], row["session"], row["structure"], row["epoch"])


def attenuation_checks(delay_pooled_rows: list[dict]) -> dict:
    """1.3(i): for every pair of widths, the ratio r(w_narrow, L) /
    r(w_wide, L) regressed on lag in seconds, session-matched (not just
    filtered independently per width, which could silently misalign
    sessions that fitted at one width but not another)."""
    by_width = {}
    for width in LAG_WIDTHS_BINS:
        by_width[width] = {
            _session_key(r): r["profile"]["lags"] for r in delay_pooled_rows
            if r["width_bins"] == width and r["profile"].get("status") == "fitted"
        }
    pairs = {}
    widths_sorted = sorted(LAG_WIDTHS_BINS)
    for i in range(len(widths_sorted)):
        for j in range(i + 1, len(widths_sorted)):
            w_narrow, w_wide = widths_sorted[i], widths_sorted[j]
            common_keys = sorted(set(by_width[w_narrow]) & set(by_width[w_wide]))
            narrow_list = [by_width[w_narrow][k] for k in common_keys]
            wide_list = [by_width[w_wide][k] for k in common_keys]
            pairs[f"w{w_narrow}_over_w{w_wide}"] = attenuation_constancy_check(narrow_list, wide_list, bin_width_s=LAG_BIN_MS / 1000.0)
    return pairs


def matched_unit_count_subsample(delay_counts_cache: dict, width: int) -> dict:
    """1.6's matched-unit-count column: every structure with at least 4
    eligible sessions is subsampled down to the common unit count (the
    minimum, across eligible structures, of the structure's own median
    unit count), refit at reduced splits/replicates for cost, and
    contrast A is reported on the subsampled data -- existence against the
    permutation null is a unit-count-sensitive statistic, and structures do
    not have the same unit count by construction (more channels are
    implanted in some structures than others)."""
    by_structure: dict[str, list[tuple]] = {}
    for key, counts in delay_counts_cache.items():
        by_structure.setdefault(key[2], []).append(key)
    unit_counts_by_structure = {s: [delay_counts_cache[k].shape[1] for k in keys] for s, keys in by_structure.items()}
    eligible = {s: c for s, c in unit_counts_by_structure.items() if len(c) >= 4}
    if not eligible:
        return {"status": "not_computed", "reason": "no structure has at least 4 delay-epoch sessions"}
    common_unit_count = int(min(np.median(c) for c in eligible.values()))
    if common_unit_count < 4:
        return {"status": "not_computed", "reason": f"common unit count across structures is {common_unit_count}, too low to fit"}

    out = {}
    for structure in sorted(eligible):
        profiles, pois, perm = [], [], []
        for key in by_structure[structure]:
            counts = delay_counts_cache[key]
            if counts.shape[1] < common_unit_count:
                continue
            rng = np.random.default_rng(_seed("matched_unit_count", *key, width))
            unit_idx = rng.choice(counts.shape[1], size=common_unit_count, replace=False)
            sub = counts[:, unit_idx, :]
            seed = _seed("matched_unit_count_lag", *key, width)
            run = _lag_run_row(sub, width, seed, n_splits=LAG_MATCHED_COUNT_N_SPLITS,
                                n_null_replicates=LAG_MATCHED_COUNT_N_NULL_REPLICATES)
            if run["profile"].get("status") != "fitted" or run["null_permutation"] is None:
                continue
            profiles.append(run["profile"]["lags"])
            pois.append(run["null_poisson"]["lags"])
            perm.append(run["null_permutation"]["lags"])
        if len(profiles) < 4:
            out[structure] = {"status": "not_enough_sessions_at_matched_unit_count", "n_sessions": len(profiles)}
            continue
        classification = classify_lag_profile(profiles, pois, perm, width_bins=width, bin_width_s=LAG_BIN_MS / 1000.0)
        out[structure] = {
            "n_sessions": len(profiles), "existence_by_lag": classification["contrast_a_existence_by_lag"],
            "n_lags_clearing_fdr": classification["n_lags_clearing_fdr"], "n_lags_tested": classification["n_lags_tested"],
        }
    return {"status": "tested", "common_unit_count": common_unit_count, "n_splits": LAG_MATCHED_COUNT_N_SPLITS,
            "n_null_replicates": LAG_MATCHED_COUNT_N_NULL_REPLICATES, "by_structure": out}


def per_structure_lag_existence(delay_rows: list[dict], delay_counts_cache: dict, width: int, white_lookup: dict) -> dict:
    """Contrast A per structure at fixed width: the lag vector of d_perm
    with FDR plus the count of clearing lags, the per-structure n, and the
    standing confound columns -- Spearman correlation of the per-session
    mean d_perm against unit count and against the session's census
    cross-validated nugget fraction (src/observability.py's nugget_fraction)
    -- plus the matched-unit-count column."""
    rows_w = [r for r in delay_rows if r["width_bins"] == width]
    structures = sorted({r["structure"] for r in rows_w})
    out = {}
    for structure in structures:
        rows_s = [r for r in rows_w if r["structure"] == structure]
        patients = {r["patient"] for r in rows_s}
        profiles, pois, perm = [], [], []
        for r in rows_s:
            if r["profile"].get("status") != "fitted" or r["null_poisson"] is None or r["null_permutation"] is None:
                continue
            profiles.append(r["profile"]["lags"])
            pois.append(r["null_poisson"]["lags"])
            perm.append(r["null_permutation"]["lags"])
        classification = classify_lag_profile(profiles, pois, perm, width_bins=width, bin_width_s=LAG_BIN_MS / 1000.0)

        d_perm_mean, unit_counts, whites = [], [], []
        for r in rows_s:
            if r["profile"].get("status") != "fitted" or r["null_permutation"] is None:
                continue
            lags, null_lags = r["profile"]["lags"], r["null_permutation"]["lags"]
            common = sorted(set(lags) & set(null_lags))
            if not common:
                continue
            d_perm_mean.append(float(np.mean([lags[k]["r_median"] - null_lags[k]["r_null_median"] for k in common])))
            unit_counts.append(r["n_units"])
            key = (r["dataset"], r["session"], r["structure"], r["epoch"], r["bin_ms"])
            whites.append(white_lookup.get(key))

        confound = {"n_sessions_with_d_perm": len(d_perm_mean)}
        if len(d_perm_mean) >= 6:
            corr_unit = spearman_permutation_test(np.array(d_perm_mean), np.array(unit_counts, dtype=float), rng=np.random.default_rng(0))
            confound["spearman_d_perm_vs_unit_count"] = {"rho": corr_unit["rho"], "p_value": corr_unit["p_value"]}
            has_white = np.array([w is not None for w in whites])
            if has_white.sum() >= 6:
                white_arr = np.array([w if w is not None else np.nan for w in whites])
                corr_white = spearman_permutation_test(
                    np.array(d_perm_mean)[has_white], white_arr[has_white], rng=np.random.default_rng(1))
                confound["spearman_d_perm_vs_nugget_fraction"] = {
                    "rho": corr_white["rho"], "p_value": corr_white["p_value"], "n": int(has_white.sum())}
            else:
                confound["spearman_d_perm_vs_nugget_fraction"] = {"status": "not_enough_matched_variance_partition_rows"}
        else:
            confound["status"] = "not_enough_sessions_for_confound_check"

        out[structure] = {
            "n_sessions": len(rows_s), "n_patients": len(patients),
            "existence_by_lag": classification["contrast_a_existence_by_lag"],
            "n_lags_clearing_fdr": classification["n_lags_clearing_fdr"], "n_lags_tested": classification["n_lags_tested"],
            "confound": confound,
        }
    out["matched_unit_count_subsample"] = matched_unit_count_subsample(delay_counts_cache, width)
    return out


def heterogeneity_on_d_perm(delay_pooled_rows: list[dict], white_lookup: dict, width: int, clearing_lags: list[int]) -> dict:
    """1.9: the same three heterogeneity correlations, re-pointed onto the
    statistic the claim now rests on -- the per-session mean of d_perm
    over the lags that cleared contrast A (1.4A) at the deciding width, not
    over every lag and not over the retired r(1) - r(3) gap effect."""
    rows_w = [r for r in delay_pooled_rows if r["width_bins"] == width]
    d_perm_mean, trial_count, unit_count, white_share = [], [], [], []
    for row in rows_w:
        if row["profile"].get("status") != "fitted" or row["null_permutation"] is None:
            continue
        lags, null_lags = row["profile"]["lags"], row["null_permutation"]["lags"]
        vals = [lags[k]["r_median"] - null_lags[k]["r_null_median"] for k in clearing_lags if k in lags and k in null_lags]
        if not vals:
            continue
        d_perm_mean.append(float(np.mean(vals)))
        trial_count.append(row["n_trials"])
        unit_count.append(row["n_units"])
        key = (row["dataset"], row["session"], row["structure"], row["epoch"], row["bin_ms"])
        white_share.append(white_lookup.get(key))

    d_perm_arr = np.array(d_perm_mean)
    trial_arr = np.array(trial_count, dtype=float)
    unit_arr = np.array(unit_count, dtype=float)
    has_white = np.array([w is not None for w in white_share])
    white_arr = np.array([w if w is not None else np.nan for w in white_share])

    if len(d_perm_arr) < 4:
        return {"status": "underpowered_by_construction", "n_sessions": len(d_perm_arr)}
    corr_trial = spearman_permutation_test(d_perm_arr, trial_arr, rng=np.random.default_rng(0))
    corr_unit = spearman_permutation_test(d_perm_arr, unit_arr, rng=np.random.default_rng(1))
    result = {
        "n_sessions": len(d_perm_arr), "width_bins": width, "clearing_lags_bins": clearing_lags,
        "spearman_d_perm_vs_trial_count": {"rho": corr_trial["rho"], "p_value": corr_trial["p_value"]},
        "spearman_d_perm_vs_unit_count": {"rho": corr_unit["rho"], "p_value": corr_unit["p_value"]},
    }
    if has_white.sum() >= 4:
        corr_white = spearman_permutation_test(d_perm_arr[has_white], white_arr[has_white], rng=np.random.default_rng(2))
        result["spearman_d_perm_vs_white_share"] = {"rho": corr_white["rho"], "p_value": corr_white["p_value"], "n": int(has_white.sum())}
    else:
        result["spearman_d_perm_vs_white_share"] = {"status": "not_enough_matched_variance_partition_rows"}
    result["explained_by_trial_count"] = bool(corr_trial["p_value"] <= 0.05 and abs(corr_trial["rho"]) >= 0.5)
    return result


def main_lag() -> None:
    """1.1(5): a window-count sweep ties window width to lag, so nothing on
    that grid can say whether r falls with lag or with narrower windows.
    This entry point untangles them -- width in bins is a fixed parameter,
    lag in bins (reported in seconds) is swept independently -- and applies
    the predeclared decision rule on the SHAPE of d_perm(lag) at the
    deciding width, w=3 bins. Writes results/state_persistence_lag.json
    beside (never in place of) results/state_persistence.json and
    results/state_persistence_profile.json, both read-only inputs here."""
    root = data_root()
    t_start = time.time()

    print("Human lag grid (widths 2,3,5 bins, both nulls, encoding+delay, 100ms)...", file=sys.stderr)
    primary_rows, delay_counts_cache = human_lag_rows(root)

    prior_path = OUTPUT_PATH  # results/state_persistence.json, a read-only input here
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    rate_matched_keep_probability = prior.get("matched_sensitivity_alm", {}).get("rate_matched_keep_probability", 1.0)
    human_window_s = EPOCH_WINDOWS_S["delay"]

    print("ALM lag arm (both nulls)...", file=sys.stderr)
    alm_rows = alm_lag_rows(root, human_window_s, rate_matched_keep_probability)
    print(f"  {len(alm_rows)} ALM lag rows, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    print("Panichello macaque prefrontal lag arm (both nulls)...", file=sys.stderr)
    panichello = panichello_lag_rows(root)
    print(f"  {len(panichello.get('rows', []))} Panichello lag rows, {time.time() - t_start:.1f}s elapsed", file=sys.stderr)

    delay_pooled = [r for r in primary_rows if r["epoch"] == "delay" and r["structure"] == "pooled"
                    and r["dataset"] in HUMAN_DATASETS]

    print("Attenuation-constancy check across widths (1.3)...", file=sys.stderr)
    attenuation = attenuation_checks(delay_pooled)

    print("Predeclared lag-shape decision rule (1.4)...", file=sys.stderr)
    decision = lag_decision(delay_pooled)
    patient_clustered = patient_clustered_human_delay(delay_pooled)
    deciding = decision["per_width"][str(LAG_DECIDING_WIDTH_BINS)]
    clearing_lags = [int(lag) for lag, c in deciding["contrast_a_existence_by_lag"].items() if c.get("fdr_significant")]

    white_lookup = _load_white_share_lookup()
    delay_rows = [r for r in primary_rows if r["epoch"] == "delay"]
    print("Per-structure existence + matched-unit-count subsample (1.6)...", file=sys.stderr)
    per_structure = per_structure_lag_existence(delay_rows, delay_counts_cache, LAG_DECIDING_WIDTH_BINS, white_lookup)

    print("Heterogeneity on the mean d_perm over 1.4A's clearing lags (1.9)...", file=sys.stderr)
    heterogeneity = heterogeneity_on_d_perm(delay_pooled, white_lookup, LAG_DECIDING_WIDTH_BINS, clearing_lags)

    print("ALM and Panichello lag-shape decisions (1.7)...", file=sys.stderr)
    alm_decision = classify_lag_profile(*_lag_lists(alm_rows, LAG_DECIDING_WIDTH_BINS),
                                         width_bins=LAG_DECIDING_WIDTH_BINS, bin_width_s=LAG_BIN_MS / 1000.0)
    panichello_decision = classify_lag_profile(*_lag_lists(panichello.get("rows", []), LAG_DECIDING_WIDTH_BINS),
                                                 width_bins=LAG_DECIDING_WIDTH_BINS, bin_width_s=LAG_BIN_MS / 1000.0)

    print("Encoding-vs-delay lag-shape discriminator (opportunistic; a full attribution verdict across "
          "species, region and task is not run here)...", file=sys.stderr)
    encoding_pooled = [r for r in primary_rows if r["epoch"] == "encoding" and r["structure"] == "pooled"
                        and r["dataset"] in HUMAN_DATASETS]
    encoding_decision = classify_lag_profile(*_lag_lists(encoding_pooled, LAG_DECIDING_WIDTH_BINS),
                                               width_bins=LAG_DECIDING_WIDTH_BINS, bin_width_s=LAG_BIN_MS / 1000.0)

    output = {
        "version": "2026-08-13",
        "relationship_to_prior_artifacts": (
            "results/state_persistence.json and results/state_persistence_profile.json are read-only inputs "
            "here and are not recomputed or edited. This file's deciding_contrast (branch field) is the "
            "current verdict on this observable's shape; where a branch or number here disagrees with either "
            "prior file, this file's is the corrected reading and the prior file's own scope/relationship "
            "field states so, per the current implementation report."
        ),
        "scope": (
            "Every (dataset, structure, session) reachable through src/corpus_sessions.iter_all_corpora; "
            "encoding and delay epochs; 100 ms bins only; width in {2, 3, 5} bins (200/300/500 ms), every "
            "lag from width to n_bins - width bins, reported in seconds. Both the Poisson and the per-unit "
            "permutation null at every (session, epoch, width) row. ALM (Inagaki) at the human delay floor "
            "window (2.3 s) with the rate-matched thinning established earlier in this project, and "
            "Panichello 2024 macaque prefrontal delay, both re-run through the identical estimator WITH "
            "BOTH NULLS -- the gap left open by the window-count profile arm. A full species/region/task "
            "attribution verdict is not run here; the ALM, Panichello, and encoding-vs-delay fields below "
            "are the data those arms need, not a completed attribution deliverable."
        ),
        "n_splits": LAG_N_SPLITS, "n_null_replicates": LAG_N_NULL_REPLICATES, "bin_width_s": LAG_BIN_MS / 1000.0,
        "widths_bins": list(LAG_WIDTHS_BINS), "deciding_width_bins": LAG_DECIDING_WIDTH_BINS,
        "attenuation_constancy_check": attenuation,
        "deciding_contrast": decision,
        "patient_clustered_human_delay": patient_clustered,
        "patient_clustered_inference_note": (
            "Existing session-row fields remain descriptive because 72 delay rows nest within 52 "
            "dataset-qualified patients. Current human F1 inference is patient_clustered_human_delay."
        ),
        "clearing_lags_bins_at_deciding_width": clearing_lags,
        "per_structure_delay": per_structure,
        "heterogeneity_on_mean_d_perm": heterogeneity,
        "alm_lag_arm": {"human_window_s": human_window_s, "rate_matched_keep_probability": rate_matched_keep_probability,
                         "decision": alm_decision, "n_rows": len(alm_rows)},
        "panichello_lag_arm": {**panichello, "rows": panichello.get("rows", []), "decision": panichello_decision},
        "encoding_vs_delay_lag_shape": {"encoding": encoding_decision, "delay": deciding},
        "human_lag_rows": primary_rows,
        "alm_lag_rows": alm_rows,
        "wall_clock_s": time.time() - t_start,
    }

    LAG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAG_OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {LAG_OUTPUT_PATH} in {time.time() - t_start:.1f}s", file=sys.stderr)
    print(json.dumps({"deciding_branch": decision["deciding_branch"], "branch_by_width": decision["branch_by_width"]}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "profile":
        main_profile()
    elif len(sys.argv) > 1 and sys.argv[1] == "lag":
        main_lag()
    elif len(sys.argv) > 1 and sys.argv[1] in {"clustered-lag", "patient-clustered-lag"}:
        main_patient_clustered_lag()
    else:
        main()
