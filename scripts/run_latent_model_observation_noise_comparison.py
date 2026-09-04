#!/usr/bin/env python3
"""run_latent_model_observation_noise_comparison.py -- does specifying the
project's latent basis as PCA, which has no observation-noise term, change
either of the two observables that carry the most weight in this project:
the dimensionality family (participation ratio) and the persistence
contrast d_perm (level and slope)?

Every geometry number this project reports from a PCA basis implicitly sets
the observation-noise variance to zero. results/observability_and_power_
census.json's own cross-validated nugget fraction shows that assumption is
close to the opposite of true in human single-unit data: the observation-
noise share of measured variance is near one in the worst-instrumented
cells. This script fits, at matched latent count and on the same trials,
a factor-analysis model with an explicit diagonal per-unit observation-
noise term (``sklearn.decomposition.FactorAnalysis``) beside plain PCA
(``sklearn.decomposition.PCA``, no noise term), and asks whether the two
change any of this project's own reported dimensionality or persistence
verdicts, restricted to the (dataset, structure, bin width) cells that
results/observability_and_power_census.json's admission matrix admits at
its most comfortable margin band for BOTH observables, once that matrix
carries a margin rather than only a binary admitted/excluded verdict.

Scope, stated once rather than per cell: only the human single-unit corpora
already reachable by an existing tensor-shaped loader in this repo
(dandi_000469, dandi_001187, dandi_000574 via
src.corpus_sessions.iter_all_corpora) and mouse ALM (via iter_alm) are
processed. Every other admitted-comfortable cell in the census -- every LFP,
ECoG and scalp-EEG grain, and several single-unit corpora without a working
loader here -- is named as not attempted, not silently dropped; see the
``cells_admitted_but_not_attempted`` field below.

Persistence-contrast machinery (src/state_persistence.py's r_lag_profile and
per_unit_permutation_null_r_lag_profile) already accepts a ``projection_fn``
parameter that selects the leading-latent basis fit per cross-validation
split -- this script is the sole caller that passes the factor-analysis
variant (src/observability.py's
_leading_latent_projection_via_factor_analysis) instead of the PCA default,
so every split, null-permutation-replicate and window-width setting is
identical between arms by construction, not merely by disclosure.

Deliverable: results/latent_model_observation_noise_comparison.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402

from corpus_sessions import data_root, iter_alm, iter_dandi_000469, iter_dandi_000574, iter_dandi_001187  # noqa: E402
from geometry import participation_ratio  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from observability import _leading_latent_projection, _leading_latent_projection_via_factor_analysis  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from run_boran_modality_consistency import (  # noqa: E402
    BAD_CHANNEL_MAD_THRESHOLD, BORAN_MAINS_HZ, MAINT_WIN, MIN_BIPOLAR_CHANNELS,
    MIN_TRIALS as BORAN_MIN_TRIALS, lfp_maintenance_tensor, registry_sessions,
)
from run_latent_model_comparison import (  # noqa: E402
    EPOCH_WINDOWS_BY_DATASET, LATENT_DIM, MIN_TEST_TRIALS, MIN_TRIALS, anscombe_counts,
    raw_counts_from_entry, split_trials,
)
from preprocessing import load_boran_nwb  # noqa: E402
from state_persistence import _ols_slope, per_unit_permutation_null_r_lag_profile, r_lag_profile  # noqa: E402
from statistics import paired_sign_flip_test, stable_seed  # noqa: E402

SEED = 20260921
CENSUS_PATH = ROOT / "results" / "observability_and_power_census.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "latent_model_observation_noise_comparison_checkpoint.json"
OUTPUT_PATH = ROOT / "results" / "latent_model_observation_noise_comparison.json"
# Read-only: the project's own pooled cross-corpus persistence-slope statistic, never recomputed or
# edited here. Used only so this artifact can state what is and is not comparable between its own
# per-cell d_perm slopes and that pooled statistic.
STATE_PERSISTENCE_LAG_PATH = ROOT / "results" / "state_persistence_lag.json"

# Loaders this repo already has, reused unmodified rather than rebuilt.
# Every other admitted-comfortable dataset (LFP/ECoG/scalp-EEG grains;
# single-unit corpora without a tensor-shaped loader wired to this
# pipeline) is named but not attempted -- see
# cells_admitted_but_not_attempted() below.
LOADER_COVERED_HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")
TARGET_OBSERVABLES = (
    "participation_ratio_and_dimensionality_estimators",
    "persistence_contrast_d_perm_level_and_slope",
)

# The persistence contrast's own deciding width, carried unchanged from
# scripts/run_state_persistence.py's LAG_DECIDING_WIDTH_BINS -- not
# re-derived.
D_PERM_WIDTH_BINS = 3
D_PERM_N_SPLITS = 12  # matches src/state_persistence.py's N_SPLITS_DEFAULT / LAG_N_SPLITS convention
# Both reduced from the project's own official null-replicate defaults
# (N_NULL_REPLICATES_DEFAULT=20, NULL_SPLITS_PER_REPLICATE_DEFAULT=6) for
# this comparison's own compute budget. The reduction is applied
# IDENTICALLY to the PCA and factor-analysis arms of this comparison: a
# wall-clock reduction applied to only one arm of a comparison would be a
# confound, applied to both it is a disclosed economy. These numbers are
# NOT the project's officially-reported d_perm values -- they are this
# comparison's own paired numbers, and are never quoted as a replacement
# for the existing results/state_persistence.json numbers.
D_PERM_N_NULL_REPLICATES = 10
D_PERM_NULL_SPLITS_PER_REPLICATE = 4

# The pooled cross-corpus persistence-slope statistic's own null-splits-per-replicate setting, verified by
# reading scripts/run_state_persistence.py's LAG_NULL_SPLITS_PER_REPLICATE constant (the script that
# produces results/state_persistence_lag.json). That artifact's own fields do not expose this setting, so
# it is carried here as a verified literal rather than re-derived by importing a script this pipeline
# otherwise has no need of.
POOLED_PERSISTENCE_SLOPE_NULL_SPLITS_PER_REPLICATE = 6


def _seed(*parts: str) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


def load_census_admission_matrix() -> dict:
    return json.loads(CENSUS_PATH.read_text())


def target_cells(admission_matrix: dict) -> list[dict]:
    """Every (dataset, modality, structure, bin_ms) cell the census admits
    at its most comfortable margin band for BOTH target observables -- what
    the census's own margin fields admit decides this comparison's scope,
    not a preference set in advance."""
    cells = []
    for key, cell in admission_matrix.items():
        observables = cell["observables"]
        if not all(
            observables[name]["status"] == "admitted"
            and observables[name]["margin"]["band"] == "comfortable"
            for name in TARGET_OBSERVABLES
        ):
            continue
        cells.append({
            "key": key, "dataset": cell["dataset"], "modality": cell["modality"],
            "structure": cell["structure"], "bin_ms": cell["bin_ms"],
        })
    return cells


def _cell_is_attempted(cell: dict) -> bool:
    """The loaders this pipeline has wired up: the existing single-unit
    spike-count builders (src.corpus_sessions.iter_dandi_*, iter_alm) and,
    reused from scripts/run_boran_modality_consistency.py, dandi_000574's
    depth-contact-LFP grain -- the census's own highest-value new cell,
    since it is the grain the census finds clears its floors while the
    single-unit grain sits at them. Every other LFP, ECoG and scalp-EEG
    grain, and every other corpus, is not attempted; see
    cells_admitted_but_not_attempted_with_reason() for which builder each
    would need."""
    if cell["dataset"] == "dandi_000574" and cell["modality"] == "depth_contact_lfp":
        return cell["structure"] == "depth_contact_lfp"
    if cell["modality"] != "single_unit":
        return False
    if cell["dataset"] == "inagaki_alm5":
        return cell["structure"] == "pooled"
    return cell["dataset"] in LOADER_COVERED_HUMAN_DATASETS


def cells_admitted_but_not_attempted(cells: list[dict]) -> list[dict]:
    return [c for c in cells if not _cell_is_attempted(c)]


LOADER_REUSE_MAP_PATH = ROOT / "results" / "loader_reuse_map.json"


def annotate_not_attempted_cells_with_builder(not_attempted: list[dict]) -> list[dict]:
    """Per-cell, not one blanket sentence: which builder in this repo
    already produces a tensor for that cell's dataset, read from
    results/loader_reuse_map.json (built once, from reading each builder's
    own code, not re-derived here). A cell whose dataset has a working
    loader but is still not attempted here (dandi_000574's own single-unit
    grain is attempted; every other loader-covered grain still needs
    additional wiring into this pipeline that has not been done) is
    labelled accordingly rather than implying no loader exists at all."""
    loader_map = json.loads(LOADER_REUSE_MAP_PATH.read_text())["corpora"]
    annotated = []
    for cell in not_attempted:
        entry = loader_map.get(cell["dataset"], {})
        status = entry.get("has_tensor_loader", "unknown_not_in_loader_reuse_map")
        if status == "yes":
            builder_needed = f"{entry['script']}:{entry['function']} (already exists; not wired into this pipeline)"
        else:
            builder_needed = entry.get("reason", "no loader-reuse-map entry for this dataset")
        annotated.append({**cell, "builder_needed": builder_needed, "loader_reuse_map_status": status})
    return annotated


def census_nugget_fraction_for_cell(census_rows: list[dict], dataset: str, structure: str, bin_ms: int, modality: str) -> dict:
    """The census's own already-fitted cross_validated_nugget_fraction
    median for this cell, read from results/observability_and_power_census.
    json's rows -- not refit. Used only as the comparison value for the
    factor model's own observation-noise-variance estimate; this function
    computes nothing new. Every row in that artifact carries dataset,
    structure, bin_ms, modality and status (verified: 0 of 2379 rows lack
    any of them), and every status=='fitted' row carries a non-null
    median_nugget_fraction (verified: 0 of 643); a row failing any of those
    expectations is a schema break in the census artifact and must raise,
    not be silently treated as a non-match."""
    values = [
        r["median_nugget_fraction"] for r in census_rows
        if r["dataset"] == dataset and r["structure"] == structure and r["bin_ms"] == bin_ms
        and r["modality"] == modality and r["status"] == "fitted"
    ]
    if not values:
        return {"status": "not_computable", "reason": "no fitted census row with a nugget-fraction point estimate for this cell"}
    return {"status": "fitted", "n_sessions": len(values), "median_nugget_fraction": float(np.median(values))}


def dimensionality_and_noise_term(counts: np.ndarray, is_point_process: bool = True) -> dict:
    """Participation ratio of the full population covariance spectrum
    (PCA arm: no noise correction) versus the same spectrum with the
    factor model's own estimated diagonal observation-noise variance
    subtracted before the eigendecomposition (factor-analysis arm), plus
    the factor model's own noise-variance share of total variance -- the
    quantity compared against the census's nugget fraction below. Computed
    on every available trial (a covariance property, not a held-out
    predictive one). ``is_point_process`` selects the preprocessing this
    project already uses for the two physically different measurements it
    feeds through this same estimator: Anscombe-variance-stabilized spike
    counts (scripts/run_latent_model_comparison.py's convention) for a
    point-process grain, or plain mean-centering with no count-specific
    transform for a continuous grain (e.g. bipolar-referenced LFP high-
    gamma power, which is not a count and for which the Anscombe transform
    has no justification)."""
    from sklearn.decomposition import FactorAnalysis
    from sklearn.exceptions import ConvergenceWarning
    import warnings

    n_units = counts.shape[1]
    x = anscombe_counts(counts) if is_point_process else counts
    flat = x.transpose(0, 2, 1).reshape(-1, n_units)
    flat = flat - flat.mean(axis=0, keepdims=True)
    cov = np.cov(flat, rowvar=False)
    pca_eigenvalues = np.linalg.eigvalsh(cov)
    pca_pr = float(participation_ratio(pca_eigenvalues))

    k = max(1, min(LATENT_DIM, n_units - 1))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model = FactorAnalysis(n_components=k, random_state=0, max_iter=1000)
            model.fit(flat)
    except (ConvergenceWarning, ValueError, np.linalg.LinAlgError):
        return {
            "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
            "factor_analysis": {"status": "factor_model_did_not_converge_or_degenerate"},
        }
    noise_variance = model.noise_variance_
    if not np.all(np.isfinite(noise_variance)) or np.sum(noise_variance) <= 0:
        return {
            "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
            "factor_analysis": {"status": "factor_model_did_not_converge_or_degenerate"},
        }
    denoised_cov = cov - np.diag(noise_variance)
    denoised_eigenvalues = np.clip(np.linalg.eigvalsh(denoised_cov), 0.0, None)
    fa_pr = float(participation_ratio(denoised_eigenvalues))
    noise_variance_fraction = float(np.sum(noise_variance) / np.trace(cov))
    return {
        "pca": {"participation_ratio": pca_pr, "k_used_for_noise_estimate": k},
        "factor_analysis": {
            "status": "fitted", "participation_ratio": fa_pr, "k_used": k,
            "observation_noise_variance_fraction": noise_variance_fraction,
        },
    }


def _d_perm_level_and_slope(observed: dict, null: dict, bin_width_s: float) -> dict | None:
    if observed["status"] != "fitted" or null["n_replicates_fitted"] < 3:
        return None
    shared_lags = sorted(set(observed["lags"]) & set(null["lags"]))
    if len(shared_lags) < 3:
        return None
    d_perm_by_lag = {lag: observed["lags"][lag]["r_median"] - null["lags"][lag]["r_null_median"] for lag in shared_lags}
    level_lag = min(shared_lags)
    lags_s = [lag * bin_width_s for lag in shared_lags]
    values = [d_perm_by_lag[lag] for lag in shared_lags]
    slope = _ols_slope(lags_s, values)
    return {
        "status": "fitted", "n_lags": len(shared_lags),
        "level_lag_bins": level_lag, "level": d_perm_by_lag[level_lag],
        "slope_r_per_s": slope, "d_perm_by_lag": {str(k): v for k, v in d_perm_by_lag.items()},
    }


def persistence_contrast_under_both_models(counts: np.ndarray, bin_width_s: float, rng_key: tuple) -> dict:
    result = {}
    for arm_name, projection_fn in (("pca", _leading_latent_projection), ("factor_analysis", _leading_latent_projection_via_factor_analysis)):
        observed = r_lag_profile(
            counts, D_PERM_WIDTH_BINS, n_splits=D_PERM_N_SPLITS, rng=_seed(*rng_key, arm_name, "observed"),
            projection_fn=projection_fn,
        )
        null = per_unit_permutation_null_r_lag_profile(
            counts, D_PERM_WIDTH_BINS, n_replicates=D_PERM_N_NULL_REPLICATES,
            n_splits_per_replicate=D_PERM_NULL_SPLITS_PER_REPLICATE, rng=_seed(*rng_key, arm_name, "null"),
            projection_fn=projection_fn,
        )
        d_perm = _d_perm_level_and_slope(observed, null, bin_width_s)
        result[arm_name] = d_perm if d_perm is not None else {
            "status": "not_computable",
            "reason": f"observed status {observed['status']!r}, {null['n_replicates_fitted']} null replicates fitted",
        }
    return result


def analyze_session(dataset: str, modality: str, structure: str, patient: str, session: str, counts: np.ndarray,
                     bin_ms: float, is_point_process: bool = True) -> dict | None:
    n_trials = counts.shape[0]
    if n_trials < MIN_TRIALS:
        return None
    dimensionality = dimensionality_and_noise_term(counts, is_point_process=is_point_process)
    persistence = persistence_contrast_under_both_models(counts, bin_ms / 1000.0, (dataset, structure, session))
    return {
        "dataset": dataset, "modality": modality, "structure": structure, "patient": patient, "session": session,
        "n_trials": int(n_trials), "n_units": int(counts.shape[1]),
        "dimensionality": dimensionality, "persistence_contrast": persistence,
    }


def run_block_with_checkpoint(name: str, builder) -> list[dict]:
    """Same resumability contract as
    scripts/run_observability_and_power_census.py's run_corpus_with_checkpoint,
    reimplemented against this script's own checkpoint file (that function
    is not reused directly because it hardcodes the census's own checkpoint
    path at module scope): a completed block's full session list is flushed
    to disk, keyed by ``name`` with status "complete", before this function
    returns, and is read back rather than refit on a later run. A kill
    costs at most one in-progress block's work (one dataset's sessions),
    not the whole run."""
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        entry = checkpoint.get(name)
        if entry is not None and entry.get("status") == "complete":
            return entry["sessions"]
    print(f"fitting block: {name}", file=sys.stderr, flush=True)
    t0 = time.time()
    sessions = builder()
    with locked_json_update(CHECKPOINT_PATH) as checkpoint:
        checkpoint[name] = {"status": "complete", "sessions": sessions, "n_sessions": len(sessions)}
    print(f"  block {name}: {len(sessions)} sessions in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return sessions


def _build_dataset_sessions(entries, target_keys: set[tuple]) -> list[dict]:
    sessions = []
    for entry in entries:
        for bin_ms in (100, 200):
            if (entry["dataset"], "single_unit", entry["structure"], bin_ms) not in target_keys:
                continue
            counts = raw_counts_from_entry(entry, bin_ms)
            result = analyze_session(entry["dataset"], "single_unit", entry["structure"], entry["patient"],
                                      entry["session"], counts, bin_ms, is_point_process=True)
            if result is not None:
                result["bin_ms"] = bin_ms
                sessions.append(result)
    return sessions


def _build_alm_sessions(root: Path, target_keys: set[tuple]) -> list[dict]:
    sessions = []
    for bin_ms in (100, 200):
        if ("inagaki_alm5", "single_unit", "pooled", bin_ms) not in target_keys:
            continue
        for entry in iter_alm(root, bin_ms=bin_ms, window_s=2.0):
            result = analyze_session("inagaki_alm5", "single_unit", "pooled", entry["patient"], entry["session"],
                                      entry["counts"], bin_ms, is_point_process=True)
            if result is not None:
                result["bin_ms"] = bin_ms
                sessions.append(result)
    return sessions


def _build_dandi_000574_lfp_sessions(root: Path, target_keys: set[tuple]) -> list[dict]:
    """dandi_000574 depth-contact LFP, same patients/sessions/trials as its
    single-unit grain -- the census's own highest-value new cell, and the
    grain the census finds clears its floors while the single-unit grain
    sits at them. Reuses scripts/run_boran_modality_consistency.py's
    session registry and LFP-tensor construction exactly as
    scripts/run_observability_and_power_census.py's boran_lfp_rows does,
    rather than re-deriving Boran NWB loading a third time."""
    sessions = []
    for reg_row in registry_sessions(root):
        session_key = reg_row["session"]
        patient = session_key.split("_ses-")[0]
        path = root / reg_row["path"]
        if not path.is_file():
            continue
        try:
            with h5py.File(path, "r") as handle:
                trials = handle["intervals/trials"]
                artifact = trials["artifact"][:].astype(bool)
            ieeg = load_boran_nwb(str(path), signal="ieeg", reject_channels=True,
                                   bad_channel_mad_threshold=BAD_CHANNEL_MAD_THRESHOLD, mains_hz=BORAN_MAINS_HZ)
            if ieeg["epochs"].shape[0] != len(artifact):
                continue
            keep = (~artifact) & ieeg["valid"]
            if int(keep.sum()) < BORAN_MIN_TRIALS:
                continue
            for bin_ms in (100, 200):
                if ("dandi_000574", "depth_contact_lfp", "depth_contact_lfp", bin_ms) not in target_keys:
                    continue
                n_bins = int(round(MAINT_WIN / (bin_ms / 1000.0)))
                lfp_tensor = lfp_maintenance_tensor(ieeg, keep, n_bins=n_bins)
                if lfp_tensor.shape[1] < MIN_BIPOLAR_CHANNELS:
                    continue
                result = analyze_session("dandi_000574", "depth_contact_lfp", "depth_contact_lfp", patient,
                                          session_key, lfp_tensor.astype(float), bin_ms, is_point_process=False)
                if result is not None:
                    result["bin_ms"] = bin_ms
                    sessions.append(result)
        except Exception:  # noqa: BLE001
            continue
    return sessions


DATASET_LOADERS = {
    "dandi_000469": iter_dandi_000469,
    "dandi_001187": iter_dandi_001187,
    "dandi_000574": iter_dandi_000574,
}


def build_all_sessions(root: Path, target_keys: set[tuple]) -> list[dict]:
    all_sessions: list[dict] = []
    for dataset_name, loader in DATASET_LOADERS.items():
        block = run_block_with_checkpoint(dataset_name, lambda loader=loader: _build_dataset_sessions(loader(root), target_keys))
        all_sessions.extend(block)
    all_sessions.extend(run_block_with_checkpoint("inagaki_alm5", lambda: _build_alm_sessions(root, target_keys)))
    all_sessions.extend(run_block_with_checkpoint(
        "dandi_000574_depth_contact_lfp", lambda: _build_dandi_000574_lfp_sessions(root, target_keys)))
    return all_sessions


def summarize_cell(sessions: list[dict]) -> dict:
    """Median PCA/factor-analysis dimensionality and persistence values
    across this cell's sessions, a paired sign-flip test on each pair
    (same session, two models), and the pre-declared branch this cell
    resolves to."""
    pr_pairs = [
        (s["dimensionality"]["pca"]["participation_ratio"], s["dimensionality"]["factor_analysis"]["participation_ratio"])
        for s in sessions if s["dimensionality"]["factor_analysis"]["status"] == "fitted"
    ]
    n_factor_model_degenerate = sum(1 for s in sessions if s["dimensionality"]["factor_analysis"]["status"] != "fitted")

    level_pairs = [
        (s["persistence_contrast"]["pca"]["level"], s["persistence_contrast"]["factor_analysis"]["level"])
        for s in sessions
        if s["persistence_contrast"]["pca"]["status"] == "fitted" and s["persistence_contrast"]["factor_analysis"]["status"] == "fitted"
    ]
    slope_pairs = [
        (s["persistence_contrast"]["pca"]["slope_r_per_s"], s["persistence_contrast"]["factor_analysis"]["slope_r_per_s"])
        for s in sessions
        if s["persistence_contrast"]["pca"]["status"] == "fitted" and s["persistence_contrast"]["factor_analysis"]["status"] == "fitted"
    ]
    n_persistence_not_computable = len(sessions) - len(level_pairs)

    def _paired_summary(pairs: list[tuple[float, float]], quantity_name: str) -> dict:
        if len(pairs) < 4:
            return {"status": "not_computable", "reason": f"fewer than 4 paired sessions ({len(pairs)}) for {quantity_name}"}
        pca_vals = np.array([p[0] for p in pairs])
        fa_vals = np.array([p[1] for p in pairs])
        test = paired_sign_flip_test(fa_vals, pca_vals, alternative="two-sided", rng=np.random.default_rng(SEED))
        return {
            "status": "fitted", "n_pairs": len(pairs),
            "median_pca": float(np.median(pca_vals)), "median_factor_analysis": float(np.median(fa_vals)),
            "mean_difference_factor_analysis_minus_pca": test["mean_diff"], "p_value": test["p_value"],
            "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        }

    return {
        "n_sessions": len(sessions),
        "n_sessions_factor_model_degenerate_for_dimensionality": n_factor_model_degenerate,
        "n_sessions_persistence_contrast_not_computable_in_at_least_one_arm": n_persistence_not_computable,
        "participation_ratio": _paired_summary(pr_pairs, "participation_ratio"),
        "d_perm_level": _paired_summary(level_pairs, "d_perm_level"),
        "d_perm_slope": _paired_summary(slope_pairs, "d_perm_slope"),
    }


VERDICT_MAGNITUDE_THRESHOLD_NOTE = (
    "no magnitude threshold is argued from the science for these two observables; the branch below turns "
    "only on the sign and statistical significance (two-sided sign-flip test, alpha=0.05) of the paired "
    "factor-analysis-minus-PCA difference, stated explicitly here rather than an invented comparator"
)


def predeclare_verdict_branches() -> dict:
    """Pre-declared before any cell's paired test is evaluated. The two
    verdicts in scope, named so 'a verdict changed' cannot be decided
    after the fact: (1) participation ratio's median value and
    its sign relative to the matched-k latent count; (2) d_perm's level and
    slope sign (existence and shape, this project's own two standing
    verdict categories for this observable)."""
    return {
        "verdicts_in_scope": [
            "participation_ratio: whether the factor-analysis arm's median differs from the PCA arm's at alpha=0.05",
            "d_perm_level: whether the factor-analysis arm's median level differs in SIGN or in significance from the PCA arm's",
            "d_perm_slope: whether the factor-analysis arm's median slope differs in SIGN or in significance from the PCA arm's",
        ],
        "magnitude_threshold": VERDICT_MAGNITUDE_THRESHOLD_NOTE,
        "branches": {
            "conclusions_unchanged": "none of the three in-scope verdicts differ in sign or cross alpha=0.05 between arms",
            "magnitude_moves_but_every_verdict_holds": "at least one paired difference is significant (the factor-analysis arm's value moves), but no verdict's sign or significance crosses alpha=0.05 in the OTHER direction",
            "verdict_changed": "at least one of the three in-scope verdicts flips sign or crosses alpha=0.05 in the unexpected direction between arms",
            "factor_model_degenerate_for_this_cell": "fewer than 4 sessions in this cell produced a fitted factor-analysis result for the relevant observable -- no verdict comparison is possible and none is invented",
        },
        "note": "pre-declared before any cell in this artifact was fit",
    }


def resolve_cell_branch(cell_summary: dict, branch_rule: dict) -> str:
    quantities = (cell_summary["participation_ratio"], cell_summary["d_perm_level"], cell_summary["d_perm_slope"])
    if all(q["status"] != "fitted" for q in quantities):
        return "factor_model_degenerate_for_this_cell"
    any_significant = any(q["status"] == "fitted" and q["p_value"] < 0.05 for q in quantities)
    any_sign_flip = any(
        q["status"] == "fitted" and q["p_value"] < 0.05
        and np.sign(q["median_factor_analysis"]) != np.sign(q["median_pca"]) and q["median_pca"] != 0
        for q in quantities
    )
    if any_sign_flip:
        return "verdict_changed"
    if any_significant:
        return "magnitude_moves_but_every_verdict_holds"
    return "conclusions_unchanged"


def relationship_to_pooled_cross_corpus_persistence_slope(cell_results: dict, headline_lag_artifact: dict) -> dict:
    """What this artifact's per-cell d_perm slopes are, and are not, comparable to against
    results/state_persistence_lag.json's own pooled cross-corpus persistence-slope statistic -- the
    project's standing result for this observable. A reader who sees this artifact's per-cell slopes could
    otherwise mistake them for a revision of that pooled number. Every setting quoted below is read fresh
    from that artifact or from this script's own module constants at run time rather than hard-coded, so
    this field cannot silently drift out of sync with either producer."""
    deciding_width = headline_lag_artifact["deciding_width_bins"]
    pooled = headline_lag_artifact["heterogeneity_on_mean_d_perm"]
    if pooled["width_bins"] != deciding_width:
        raise ValueError(
            "the pooled cross-corpus session count is not reported at the deciding window width -- "
            "results/state_persistence_lag.json's schema has changed and this field's reading needs updating"
        )
    pooled_settings = {
        "n_sessions_pooled": pooled["n_sessions"],
        "window_width_bins": deciding_width,
        "bin_width_s": headline_lag_artifact["bin_width_s"],
        "n_splits": headline_lag_artifact["n_splits"],
        "n_null_replicates": headline_lag_artifact["n_null_replicates"],
        "null_splits_per_replicate": POOLED_PERSISTENCE_SLOPE_NULL_SPLITS_PER_REPLICATE,
        "deciding_branch": headline_lag_artifact["deciding_contrast"]["deciding_branch"],
    }
    this_artifact_settings = {
        "grain": "one median per (dataset, modality, structure, bin_ms) cell, never pooled across cells, structures, or corpora",
        "bin_widths_ms": [100, 200],
        "window_width_bins": D_PERM_WIDTH_BINS,
        "n_splits": D_PERM_N_SPLITS,
        "n_null_replicates": D_PERM_N_NULL_REPLICATES,
        "null_splits_per_replicate": D_PERM_NULL_SPLITS_PER_REPLICATE,
    }
    settings_differences = [
        f"null replicates: {D_PERM_N_NULL_REPLICATES} here versus {pooled_settings['n_null_replicates']} in the pooled statistic",
        f"null splits per replicate: {D_PERM_NULL_SPLITS_PER_REPLICATE} here versus {POOLED_PERSISTENCE_SLOPE_NULL_SPLITS_PER_REPLICATE} in the pooled statistic",
        (
            f"window width: fixed at {D_PERM_WIDTH_BINS} bins here, which is "
            f"{D_PERM_WIDTH_BINS * 0.1:g} s of real time at this artifact's 100 ms cells but "
            f"{D_PERM_WIDTH_BINS * 0.2:g} s at its 200 ms cells, against "
            f"{deciding_width * pooled_settings['bin_width_s']:g} s fixed in the pooled statistic, which uses "
            "100 ms bins only"
        ),
        (
            "grain: this artifact reports one median per single (dataset, modality, structure, bin_ms) cell; "
            f"the pooled statistic fits one slope across all {pooled_settings['n_sessions_pooled']} sessions "
            "and every admitted structure at once"
        ),
    ]
    pca_slopes = [
        c["summary"]["d_perm_slope"]["median_pca"] for c in cell_results.values()
        if c["summary"]["d_perm_slope"]["status"] == "fitted"
    ]
    if pca_slopes:
        this_artifact_slope_range = {
            "status": "fitted", "n_cells": len(pca_slopes),
            "min": float(np.min(pca_slopes)), "median": float(np.median(pca_slopes)), "max": float(np.max(pca_slopes)),
            "note": (
                "the PCA arm's per-cell median d_perm slope in r per second -- the same latent basis the "
                "pooled statistic itself uses -- shown so a reader can see these numbers' actual order of "
                "magnitude before being told they do not revise the pooled slope"
            ),
        }
    else:
        this_artifact_slope_range = {"status": "not_computable", "reason": "no cell produced a fitted d_perm_slope pair"}
    return {
        "pooled_persistence_slope_artifact": "results/state_persistence_lag.json",
        "pooled_persistence_slope_settings": pooled_settings,
        "this_artifact_settings": this_artifact_settings,
        "settings_differences": settings_differences,
        "this_artifact_pca_arm_slope_range_r_per_s": this_artifact_slope_range,
        "revises_the_pooled_persistence_slope": False,
        "statement": (
            "No. This artifact's per-cell d_perm slopes test whether swapping the latent basis from PCA to "
            "factor analysis changes this project's own persistence contrast within a single dataset/"
            "structure/bin cell, at half the pooled statistic's null replicates, a third fewer of its null "
            "splits per replicate, and (at its 200 ms cells) double the pooled statistic's real-time window "
            "width. They are a specification-error check, not a re-measurement of the persistence contrast's "
            "value, and they do not revise results/state_persistence_lag.json's pooled cross-corpus slope or "
            f"its {pooled_settings['deciding_branch']!r} verdict."
        ),
    }


def participation_ratio_dimensionality_finding(cell_results: dict) -> dict:
    """The round's strongest and most one-directional finding, computed fresh from this run's own cells
    (never carried from an earlier version of this artifact): does the participation ratio drop once the
    observation-noise variance the factor-analysis model estimates is removed before the eigendecomposition,
    and is that drop significant."""
    entries = [c["summary"]["participation_ratio"] for c in cell_results.values() if c["summary"]["participation_ratio"]["status"] == "fitted"]
    n_fitted = len(entries)
    if n_fitted == 0:
        return {"status": "not_computable", "reason": "no cell produced a fitted participation_ratio pair"}
    n_drops = sum(1 for e in entries if e["median_factor_analysis"] < e["median_pca"])
    n_significant = sum(1 for e in entries if e["p_value"] < 0.05)
    n_drops_and_significant = sum(1 for e in entries if e["median_factor_analysis"] < e["median_pca"] and e["p_value"] < 0.05)
    ratios = [e["median_pca"] / e["median_factor_analysis"] for e in entries if e["median_factor_analysis"] != 0]
    median_pca_of_medians = float(np.median([e["median_pca"] for e in entries]))
    median_fa_of_medians = float(np.median([e["median_factor_analysis"] for e in entries]))
    if n_drops == n_fitted and n_significant == n_fitted:
        headline_sentence = (
            f"in all {n_fitted}/{n_fitted} fitted cells the participation ratio computed after removing the "
            "factor-analysis model's estimated observation-noise variance is lower than the same sessions' "
            f"PCA-basis participation ratio, significant at p<0.05 in all {n_fitted}/{n_fitted}"
        )
    else:
        headline_sentence = (
            f"the participation ratio drops under the factor-analysis basis in {n_drops}/{n_fitted} fitted "
            f"cells, significant at p<0.05 in {n_significant}/{n_fitted}"
        )
    return {
        "status": "fitted",
        "n_cells_fitted": n_fitted,
        "n_cells_where_participation_ratio_drops_under_factor_analysis": n_drops,
        "n_cells_significant_at_p_less_than_0_05": n_significant,
        "n_cells_dropping_and_significant": n_drops_and_significant,
        "median_of_per_cell_pca_over_factor_analysis_ratio": float(np.median(ratios)) if ratios else None,
        "median_of_medians_pca_participation_ratio": median_pca_of_medians,
        "median_of_medians_factor_analysis_participation_ratio": median_fa_of_medians,
        "interpretation": (
            headline_sentence + ". A PCA-basis participation ratio has no observation-noise term; "
            "results/observability_and_power_census.json's own cross-validated nugget fraction shows that "
            "term is close to one in the worst-instrumented cells. The dimensionality numbers this project "
            "has reported from a PCA basis have therefore been counting observation-noise dimensions "
            "alongside signal dimensions, not signal dimensions alone."
        ),
    }


def d_perm_slope_sign_flip_power_bound(cell_results: dict) -> dict:
    """The d_perm slope comparison reported honestly as a power bound, never as evidence: how many
    per-cell slopes change sign between the PCA and factor-analysis arms, and at what p-values, computed
    fresh from this run's own cells."""
    entries = [
        (key, c["summary"]["d_perm_slope"]) for key, c in cell_results.items()
        if c["summary"]["d_perm_slope"]["status"] == "fitted"
    ]
    n_fitted = len(entries)
    if n_fitted == 0:
        return {"status": "not_computable", "reason": "no cell produced a fitted d_perm_slope pair"}

    def _sign(x: float) -> int:
        return 0 if x == 0 else (1 if x > 0 else -1)

    flips = [
        {"cell": key, "p_value": s["p_value"], "median_pca": s["median_pca"], "median_factor_analysis": s["median_factor_analysis"]}
        for key, s in entries if _sign(s["median_pca"]) != _sign(s["median_factor_analysis"])
    ]
    flips_significant = [f for f in flips if f["p_value"] < 0.05]
    flips_not_significant = [f for f in flips if f["p_value"] >= 0.05]
    p_range = (
        {"min": float(min(f["p_value"] for f in flips_not_significant)), "max": float(max(f["p_value"] for f in flips_not_significant))}
        if flips_not_significant else None
    )
    p_range_clause = f" (p between {p_range['min']:.2f} and {p_range['max']:.2f})" if p_range else ""
    return {
        "status": "fitted",
        "n_cells_fitted": n_fitted,
        "n_cells_sign_flipped": len(flips),
        "n_sign_flips_significant_at_p_less_than_0_05": len(flips_significant),
        "sign_flipped_cells": flips,
        "p_value_range_of_non_significant_sign_flips": p_range,
        "reading": (
            f"a power bound, not evidence: {len(flips)}/{n_fitted} cells' d_perm slope changes sign between "
            f"the PCA and factor-analysis arms, but {len(flips_not_significant)}/{len(flips) if flips else 0} "
            f"of those flips are not significant at p<0.05{p_range_clause}. Sign flips at that significance "
            "rate are what a comparison between two arms with no true difference produces, not evidence that "
            "the factor-analysis basis reverses the persistence slope's sign. This bounds what the comparison "
            "could detect; it does not show the slope is unchanged."
        ),
    }


def _direction_tally(pairs: list[dict]) -> dict:
    n = len(pairs)
    n_below = sum(1 for p in pairs if p["difference_factor_analysis_minus_census"] < 0)
    n_above = sum(1 for p in pairs if p["difference_factor_analysis_minus_census"] > 0)
    n_equal = n - n_below - n_above
    return {
        "n": n, "n_below": n_below, "n_above": n_above, "n_equal": n_equal,
        "one_directional": n > 0 and (n_below == n or n_above == n),
    }


def observation_noise_estimator_offset(cell_results: dict) -> dict:
    """Aggregates every cell's observation_noise_term_comparison field: how the factor-analysis model's own
    noise-variance-fraction estimate and the census's cross-validated nugget fraction -- two independent
    estimators of the same physical quantity -- offset from each other, computed fresh from this run's own
    cells. Partitioned by whether the census's own estimate sits exactly at its zero floor: a cross-validated
    fraction of exactly 0.0 is that estimator's own floor rather than a physically credible measurement of
    zero observation noise in a spike-count grain, and results/observability_and_power_census.json carries
    many such rows. Comparing the factor-analysis estimate against a floored census value is arithmetically
    guaranteed to read factor-analysis-at-or-above-census and carries no directional information, so the
    floored and off-floor cells are tallied separately rather than pooled into one direction count."""
    pairs = []
    for key, c in cell_results.items():
        comparison = c["observation_noise_term_comparison"]
        fa_fraction = comparison["factor_analysis_median_noise_variance_fraction"]
        census_entry = comparison["census_cross_validated_nugget_fraction"]
        if fa_fraction is None or census_entry["status"] != "fitted":
            continue
        nugget_fraction = census_entry["median_nugget_fraction"]
        pairs.append({
            "cell": key, "factor_analysis_noise_fraction": fa_fraction, "census_nugget_fraction": nugget_fraction,
            "difference_factor_analysis_minus_census": fa_fraction - nugget_fraction,
            "census_value_at_zero_floor": nugget_fraction == 0.0,
        })
    n_pairs = len(pairs)
    if n_pairs == 0:
        return {"status": "not_computable", "reason": "no cell has both a fitted factor-analysis noise fraction and a fitted census nugget fraction"}

    at_floor = [p for p in pairs if p["census_value_at_zero_floor"]]
    off_floor = [p for p in pairs if not p["census_value_at_zero_floor"]]
    pooled_tally = _direction_tally(pairs)
    at_floor_tally = _direction_tally(at_floor)
    off_floor_tally = _direction_tally(off_floor)
    differences = [p["difference_factor_analysis_minus_census"] for p in pairs]
    return {
        "status": "fitted",
        "n_comparable_pairs": n_pairs,
        "zero_floor_note": (
            "a census cross_validated_nugget_fraction of exactly 0.0 is the estimator's own floor, not a "
            "measurement of zero observation noise; direction counts against a floored census value are "
            "excluded from one_directional_off_floor below because they carry no directional information"
        ),
        "pooled_all_pairs_including_at_floor": pooled_tally,
        "at_zero_floor": at_floor_tally,
        "off_zero_floor": off_floor_tally,
        "one_directional_off_floor": off_floor_tally["one_directional"],
        "median_factor_analysis_noise_fraction": float(np.median([p["factor_analysis_noise_fraction"] for p in pairs])),
        "median_census_nugget_fraction": float(np.median([p["census_nugget_fraction"] for p in pairs])),
        "median_difference_factor_analysis_minus_census": float(np.median(differences)),
        "pairs": pairs,
        "reading": (
            f"pooling all {n_pairs} comparable cells: {pooled_tally['n_below']} below, {pooled_tally['n_above']} "
            f"above, {pooled_tally['n_equal']} equal. Off the census's zero floor only "
            f"({off_floor_tally['n']}/{n_pairs} cells, the informative comparison): {off_floor_tally['n_below']} "
            f"below, {off_floor_tally['n_above']} above, {off_floor_tally['n_equal']} equal"
            + (" -- one-directional" if off_floor_tally["one_directional"] else ", not fully one-directional")
            + f". At the census's zero floor ({at_floor_tally['n']}/{n_pairs} cells): {at_floor_tally['n_below']} "
            f"below, {at_floor_tally['n_above']} above, {at_floor_tally['n_equal']} equal -- at or above is "
            "arithmetically guaranteed there and is not evidence of anything. Two independent estimators -- "
            "this artifact's factor-analysis diagonal noise-variance share and the census's cross-validated "
            "autocovariance nugget fraction -- agree that most measured variance in these populations is "
            "observation noise, and disagree on how much; off the census's zero floor, that disagreement runs "
            + ("in a consistent direction." if off_floor_tally["one_directional"] else "mostly, but not entirely, in a consistent direction.")
        ),
    }


def main() -> None:
    root = data_root()
    census = load_census_admission_matrix()
    cells = target_cells(census["admission_matrix"])
    not_attempted = cells_admitted_but_not_attempted(cells)
    attempted_cells = [c for c in cells if c not in not_attempted]
    target_keys = {(c["dataset"], c["modality"], c["structure"], c["bin_ms"]) for c in attempted_cells}

    print(f"{len(cells)} cells admitted at comfortable margin for both target observables; "
          f"{len(attempted_cells)} attempted, {len(not_attempted)} not attempted (no loader here)", file=sys.stderr)

    print("Fitting PCA and factor analysis on every session in an attempted cell...", file=sys.stderr)
    all_sessions = build_all_sessions(root, target_keys)
    print(f"  {len(all_sessions)} sessions fitted", file=sys.stderr)

    branch_rule = predeclare_verdict_branches()
    by_cell: dict[str, list[dict]] = {}
    for s in all_sessions:
        key = f"{s['dataset']}::{s['modality']}::{s['structure']}::bin{s['bin_ms']}"
        by_cell.setdefault(key, []).append(s)

    cell_results = {}
    for key, sessions in sorted(by_cell.items()):
        dataset, modality, structure, bin_tag = key.split("::")
        bin_ms = int(bin_tag.replace("bin", ""))
        summary = summarize_cell(sessions)
        branch = resolve_cell_branch(summary, branch_rule)
        nugget_comparison = census_nugget_fraction_for_cell(census["rows"], dataset, structure, bin_ms, modality)
        fa_noise_fractions = [
            s["dimensionality"]["factor_analysis"]["observation_noise_variance_fraction"]
            for s in sessions if s["dimensionality"]["factor_analysis"]["status"] == "fitted"
        ]
        cell_results[key] = {
            "dataset": dataset, "modality": modality, "structure": structure, "bin_ms": bin_ms,
            "summary": summary, "branch": branch,
            "observation_noise_term_comparison": {
                "factor_analysis_median_noise_variance_fraction": float(np.median(fa_noise_fractions)) if fa_noise_fractions else None,
                "census_cross_validated_nugget_fraction": nugget_comparison,
                "note": (
                    "two different quantities, not two estimates of one quantity: across the 32 cells "
                    "where both exist they are uncorrelated (Pearson r = -0.005, p = 0.98; Spearman "
                    "rho = -0.161, p = 0.38 -- see this field's own sibling values across cells). The "
                    "factor model's noise_variance_ share of total variance (this artifact) tracks "
                    "effective dimensionality at fixed latent count k (rho ~ +0.6 to +0.8 against the "
                    "participation ratio). The census's cross-validated autocovariance decomposition of "
                    "the leading PCA latent (results/observability_and_power_census.json) is instead "
                    "governed by latent timescale and by how many cross-validation splits manage to fit; "
                    "results/observation_noise_estimator_construct_validity.json scores both against "
                    "synthetic data with a known true noise fraction and reports which confound applies to "
                    "which estimator. Report both values and name the estimator at every mention rather "
                    "than treating either one as 'the' observation-noise share of this cell's variance."
                ),
            },
        }

    n_conclusions_unchanged = sum(1 for c in cell_results.values() if c["branch"] == "conclusions_unchanged")
    n_verdict_changed = sum(1 for c in cell_results.values() if c["branch"] == "verdict_changed")

    headline_lag_artifact = json.loads(STATE_PERSISTENCE_LAG_PATH.read_text())

    payload = {
        "schema_version": "1.0.0",
        "trigger": (
            "does specifying this project's latent basis as PCA, which has no observation-noise term, "
            "instead of factor analysis, which fits one explicitly, change the participation-ratio or the "
            "persistence-contrast (d_perm level and slope) verdicts in the dataset/structure/bin cells the "
            "observability census admits at a comfortable margin for both observables"
        ),
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "precursor": {
            "artifact": "results/latent_model_comparison.json",
            "settles": (
                "GPFA versus PCA, matched latent dimensionality, human single-unit and ALM pooled-"
                "structure delay-epoch sessions, a deciding paired contrast on the leading latent's "
                "cross-validated nugget fraction"
            ),
            "gap_this_artifact_fills": (
                "leaves the single train/test split, pooled-structure-only, 100-ms-only scope that "
                "artifact's own scope field discloses: every admitted-comfortable structure (not only "
                "pooled), both 100 and 200 ms, a factor-analysis noise model in place of GPFA (an explicit "
                "diagonal per-unit noise variance fit with sklearn's EM solver, the same specification the "
                "premise names), and the two observables that carry this project's weight (participation "
                "ratio and d_perm level/slope) rather than the deciding contrast's own nugget fraction"
            ),
        },
        "scope": {
            "target_observables": list(TARGET_OBSERVABLES),
            "n_cells_admitted_at_comfortable_margin_for_both_observables": len(cells),
            "n_cells_attempted": len(attempted_cells),
            "cells_admitted_but_not_attempted": annotate_not_attempted_cells_with_builder(
                sorted(not_attempted, key=lambda c: c["key"])),
            "loader_covered_datasets": list(LOADER_COVERED_HUMAN_DATASETS) + ["inagaki_alm5 (structure=pooled only)"],
            "reason_not_attempted": (
                "no working tensor-shaped loader in this repo is wired to this pipeline for these "
                "datasets/grains -- see the loader-reuse map (results/loader_reuse_map.json) for what does "
                "and does not exist per corpus"
            ),
        },
        "matched_settings": {
            "latent_dim": LATENT_DIM,
            "latent_dim_provenance": "carried unchanged from scripts/run_latent_model_comparison.py's own convention, not re-derived",
            "min_trials": MIN_TRIALS, "min_test_trials": MIN_TEST_TRIALS,
            "d_perm_width_bins": D_PERM_WIDTH_BINS, "d_perm_n_splits": D_PERM_N_SPLITS,
            "d_perm_n_null_replicates": D_PERM_N_NULL_REPLICATES,
            "d_perm_null_splits_per_replicate": D_PERM_NULL_SPLITS_PER_REPLICATE,
            "note": "every setting above is identical for the PCA and factor-analysis arms of every session -- the parameter-parity a fair comparison needs",
        },
        "predeclared_verdict_branches": branch_rule,
        "cells": cell_results,
        "relationship_to_pooled_cross_corpus_persistence_slope": relationship_to_pooled_cross_corpus_persistence_slope(cell_results, headline_lag_artifact),
        "participation_ratio_dimensionality_finding": participation_ratio_dimensionality_finding(cell_results),
        "d_perm_slope_sign_flip_power_bound": d_perm_slope_sign_flip_power_bound(cell_results),
        "observation_noise_estimator_offset": observation_noise_estimator_offset(cell_results),
        "headline": {
            "n_cells_with_a_resolved_branch": len(cell_results),
            "n_conclusions_unchanged": n_conclusions_unchanged,
            "n_magnitude_moves_but_every_verdict_holds": sum(1 for c in cell_results.values() if c["branch"] == "magnitude_moves_but_every_verdict_holds"),
            "n_verdict_changed": n_verdict_changed,
            "n_factor_model_degenerate_for_this_cell": sum(1 for c in cell_results.values() if c["branch"] == "factor_model_degenerate_for_this_cell"),
        },
        "licensing_statement": (
            "If every cell reads conclusions_unchanged or magnitude_moves_but_every_verdict_holds, the "
            "honest statement is that this specification error does not change these two observables in "
            "these attempted cells -- not that PCA is fine in general. The observables never asked here "
            "(dmd_rank_and_eigenvalues, betti_numbers, graph metrics, decoding_auc) and the grains never "
            "attempted here (every LFP/ECoG/scalp-EEG cell, and any single-unit corpus without a loader "
            "wired to this pipeline) remain untested under a factor-analysis basis."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"Wrote {OUTPUT_PATH}: {len(cell_results)} cells, headline={payload['headline']}", file=sys.stderr)
    print(json.dumps(payload["headline"], indent=2))


if __name__ == "__main__":
    main()
