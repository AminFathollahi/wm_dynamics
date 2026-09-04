#!/usr/bin/env python3
"""The matched-draw replication for the structure-identifiability outcome:
for every pair of anatomical structures co-recorded in the same DANDI 000469
patient, repeatedly draw unit subsamples jointly
matched on unit count AND mean firing rate (unit count via the target of the
smaller structure; rate via choosing, among many cheap candidate subsets,
the one whose mean rate is closest to the other structure's observed mean
rate -- trial count is already naturally matched within a patient, since
both structures share the same session's trials), refit the existing
per-fold pipeline unchanged (`run_human_drift_spine_000469.analyze_session`
with `unit_indices`), and report the fraction of draws with an identified
lambda per structure -- the outcome variable this project's identifiability
finding is actually about, not the lambda value itself.

This REPLACES the matched-lambda-difference design in
run_unit_count_matched_sensitivity.py (which conditions on identifiability
before comparing lambda and dead-ends when one arm is essentially never
identified -- see crack register:
matched_count_design_cannot_identify_paired_lambda_difference). That script
is not modified or restarted here.

SCOPE, DECLARED BEFORE RUNNING: DANDI 000469 only. A single matched draw
costs ~35 s (one full 5-fold CV PCA + Gaussian state-space refit) --
measured directly on this machine before choosing N_DRAWS_PER_ARM below.
Extending this same joint-matching procedure to DANDI 001187/000673 and
000574/Boran requires adding unit-index subsampling support to their own
session-fit functions (`fit_load_confinement`,
run_human_drift_spine_000574.analyze_session), which neither currently has --
that is additional engineering, not yet done, and is filed as a
crack rather than silently skipped.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci, paired_sign_flip_test  # noqa: E402
from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS, filter_units_by_region, load_spike_times,
    low_rate_unit_mask, resolve_unit_regions, unit_mean_firing_rates,
)
import run_human_drift_spine_000469 as spine469  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "structure_identifiability_matched_draws.json"
SEED = 20260807
N_DRAWS_PER_ARM = 20  # reduced from the 200 used for a single pair last round (see docstring)
N_RATE_CANDIDATES = 40  # cheap candidate subsets evaluated per draw to pick the closest-rate one
N_WORKERS = 28


def _patient_region_units(path: Path, region: str, window_s: float) -> tuple[list[np.ndarray], np.ndarray]:
    with h5py.File(path, "r") as handle:
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]
        trials = handle["intervals/trials"]
        onsets = trials["timestamps_Maintenance"][:]
    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    mask = low_rate_unit_mask(spike_lists, onsets, window_s)
    return [s for s, k in zip(spike_lists, mask) if k], onsets


def _closest_rate_subset(unit_rates: np.ndarray, target_count: int, target_rate: float,
                          rng: np.random.Generator, n_candidates: int) -> np.ndarray:
    n_available = len(unit_rates)
    best_indices, best_gap = None, np.inf
    for _ in range(n_candidates):
        candidate = rng.choice(n_available, size=target_count, replace=False)
        gap = abs(float(np.mean(unit_rates[candidate])) - target_rate)
        if gap < best_gap:
            best_gap, best_indices = gap, candidate
    return np.sort(best_indices)


def _draw_one(args: tuple) -> dict:
    (path_str, region, target_count, target_rate, window_s, base_seed, draw_index) = args
    path = Path(path_str)
    rng = np.random.default_rng(base_seed + draw_index)
    spike_lists, onsets = _patient_region_units(path, region, window_s)
    if len(spike_lists) < target_count:
        return {"draw": draw_index, "identified": False, "status": "insufficient_units_after_qc"}
    unit_rates = unit_mean_firing_rates(spike_lists, onsets, window_s)
    indices = _closest_rate_subset(unit_rates, target_count, target_rate, rng, N_RATE_CANDIDATES)
    achieved_rate = float(np.mean(unit_rates[indices]))
    fit = spine469.analyze_session(path, seed=base_seed + draw_index, region=region, unit_indices=indices)
    if fit.get("status") != "complete":
        return {"draw": draw_index, "identified": False, "status": fit.get("status"), "achieved_rate_hz": achieved_rate}
    lam = fit["summary"].get("state_space_lambda_identified_mean")
    return {
        "draw": draw_index, "identified": lam is not None, "lambda_rate": lam,
        "status": "complete", "achieved_rate_hz": achieved_rate,
    }


def _stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) ^ SEED


def _patient_path(patient: str) -> Path:
    return spine469.data_directory() / patient / f"{patient}_ses-2_ecephys+image.nwb"


def _target_rate_and_count(other_region_units: list[np.ndarray], other_onsets: np.ndarray, window_s: float,
                            n_units_self: int) -> tuple[int, float]:
    other_rates = unit_mean_firing_rates(other_region_units, other_onsets, window_s)
    return min(n_units_self, len(other_region_units)), float(np.mean(other_rates))


def main() -> None:
    artifact = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())
    window_s = 2.3
    complete_patients = {
        region: sorted(p for p, row in artifact["regions"][region]["sessions"].items() if row.get("status") == "complete")
        for region in ANATOMICAL_REGIONS
    }

    all_jobs = []
    pair_patient_meta: dict[tuple, dict] = {}

    for region_a, region_b in itertools.combinations(ANATOMICAL_REGIONS, 2):
        shared = sorted(set(complete_patients[region_a]) & set(complete_patients[region_b]))
        for patient in shared:
            path = _patient_path(patient)
            units_a, onsets_a = _patient_region_units(path, region_a, window_s)
            units_b, onsets_b = _patient_region_units(path, region_b, window_s)
            if not units_a or not units_b:
                continue
            target_count_a, target_rate_a = _target_rate_and_count(units_b, onsets_b, window_s, len(units_a))
            target_count_b, target_rate_b = _target_rate_and_count(units_a, onsets_a, window_s, len(units_b))
            base_seed_a = _stable_seed(region_a, region_b, patient, "a")
            base_seed_b = _stable_seed(region_a, region_b, patient, "b")
            start_a = len(all_jobs)
            for d in range(N_DRAWS_PER_ARM):
                all_jobs.append((str(path), region_a, target_count_a, target_rate_a, window_s, base_seed_a, d))
            start_b = len(all_jobs)
            for d in range(N_DRAWS_PER_ARM):
                all_jobs.append((str(path), region_b, target_count_b, target_rate_b, window_s, base_seed_b, d))
            pair_patient_meta[(region_a, region_b, patient)] = {
                "target_count_a": target_count_a, "target_rate_a": target_rate_a,
                "target_count_b": target_count_b, "target_rate_b": target_rate_b,
                "range_a": (start_a, start_a + N_DRAWS_PER_ARM),
                "range_b": (start_b, start_b + N_DRAWS_PER_ARM),
            }

    n_jobs = len(all_jobs)
    print(f"total matched-draw refits queued: {n_jobs}", flush=True)
    import time
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        results = pool.map(_draw_one, all_jobs)
    wall_clock_s = time.time() - t0

    pairs_out: dict[str, dict] = {}
    for (region_a, region_b, patient), meta in pair_patient_meta.items():
        draws_a = results[meta["range_a"][0]:meta["range_a"][1]]
        draws_b = results[meta["range_b"][0]:meta["range_b"][1]]
        frac_a = sum(1 for r in draws_a if r["identified"]) / len(draws_a)
        frac_b = sum(1 for r in draws_b if r["identified"]) / len(draws_b)
        pair_key = f"{region_a}__vs__{region_b}"
        pairs_out.setdefault(pair_key, {"region_a": region_a, "region_b": region_b, "patients": {}})
        pairs_out[pair_key]["patients"][patient] = {
            "target_count": meta["target_count_a"],
            "fraction_identified_a": frac_a, "fraction_identified_b": frac_b,
            "fraction_identified_diff_a_minus_b": frac_a - frac_b,
            "target_rate_hz_a": meta["target_rate_a"], "target_rate_hz_b": meta["target_rate_b"],
            "achieved_mean_rate_hz_a": float(np.mean([r.get("achieved_rate_hz") for r in draws_a if r.get("achieved_rate_hz") is not None]) or np.nan),
            "achieved_mean_rate_hz_b": float(np.mean([r.get("achieved_rate_hz") for r in draws_b if r.get("achieved_rate_hz") is not None]) or np.nan),
            "n_draws_per_arm": N_DRAWS_PER_ARM,
        }

    rng = np.random.default_rng(SEED)
    for pair_key, block in pairs_out.items():
        diffs = np.array([p["fraction_identified_diff_a_minus_b"] for p in block["patients"].values()])
        patient_ids = sorted(block["patients"])
        if len(diffs) < 2:
            block["paired_summary"] = {
                "status": "non_identified", "n_paired_patients": int(len(diffs)),
                "reason": f"fewer than 2 paired patients ({len(diffs)})",
            }
            continue
        _, lo, hi = bootstrap_ci(diffs, np.mean, n_boot=5000, rng=rng)
        sign_flip_p = paired_sign_flip_test(diffs, np.zeros_like(diffs), n_perm=10000, alternative="two-sided", rng=rng)
        block["paired_summary"] = {
            "status": "estimable", "n_paired_patients": int(len(diffs)), "patient_ids": patient_ids,
            "mean_fraction_identified_diff": float(np.mean(diffs)),
            "patient_bootstrap_ci95": [float(lo), float(hi)],
            "sign_flip_p_value": sign_flip_p.get("p_value") if isinstance(sign_flip_p, dict) else sign_flip_p,
            "interval_excludes_zero": bool(lo > 0 or hi < 0),
        }

    output = {
        "schema_version": "1.0.0", "analysis_id": "structure_identifiability_matched_draws",
        "trigger": "whether identifiability differences across structures hold across patients when jointly matched on unit count and rate",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "scope": (
            "DANDI 000469 only, every ordered pair among the 5 co-recorded structures "
            "(ANATOMICAL_REGIONS), every patient with both structures 'complete' in "
            "region_stratified_drift_000469.json. 001187/000673 and 000574/Boran are NOT covered "
            "-- their session-fit functions have no unit_indices subsampling support, which is "
            "additional engineering not yet completed (crack register: "
            "structure_identifiability_matched_draws_scope_limited_to_000469)."
        ),
        "reduced_scope_declaration": (
            f"N_DRAWS_PER_ARM={N_DRAWS_PER_ARM}, reduced from the 200 used for a single pair last "
            "round: a single matched draw costs ~35s (one full 5-fold CV PCA + Gaussian "
            "state-space refit, measured directly before choosing this count). The reduction is "
            "declared here, before running, per the standing house rule on infeasible mandatory "
            "sensitivities."
        ),
        "n_total_refits": n_jobs,
        "measured_wall_clock_s": wall_clock_s,
        "n_workers": N_WORKERS,
        "matching_method": (
            "unit count: target = min(n_units_A, n_units_B). trial count: already naturally "
            "matched within-patient (same session's trials feed both structures). rate: among "
            f"{N_RATE_CANDIDATES} cheap candidate random subsets of the target unit count, the one "
            "whose mean firing rate is closest to the OTHER structure's observed mean rate is "
            "selected for the (single, expensive) refit -- not full rejection sampling."
        ),
        "pairs": pairs_out,
    }
    OUTPUT_PATH.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(OUTPUT_PATH), "n_jobs": n_jobs, "wall_clock_s": wall_clock_s,
        "n_pairs": len(pairs_out),
    }, indent=2))


if __name__ == "__main__":
    main()
