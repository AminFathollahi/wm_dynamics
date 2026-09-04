#!/usr/bin/env python3
"""The blocking unit-count-matched sensitivity that
gates the pre-SMA-vs-hippocampus confinement-rate ordering.

PREDECLARED DECISION, written into this file before the
first run: "The regional confinement-rate ordering is supported only if, at
matched unit counts and within patient, the pre-SMA-minus-hippocampus
difference in lambda retains a patient-bootstrap interval excluding zero AND
keeps the sign it has at unmatched counts. If the sign flips or the interval
crosses zero at matched counts, the ordering is an identifiability artifact
and is reported as such." UNMATCHED_SIGN below is fixed from
results/region_stratified_drift_000469.json's corrected
ordering (pre_sma median 1.9735 s^-1 > hippocampus median 1.7295 s^-1,
i.e. positive) and is not re-derived from this script's own output.

Scope: DANDI 000469, pre-SMA vs hippocampus -- the specific pair identified as the deciding
contrast for the medial-frontal-faster-than-MTL headline. Downsampling machinery (draw_and_fit/summarize_draws) is written
generically over (path, region, target_count); extending to every ordered
structure pair in every eligible dataset
is deferred and reported as such -- R>=200 draws per (patient, structure)
already means ~8000+ full per-fold refits for this one pair alone.

Checkpointed per patient: safe to kill and resume (re-run this script; it
skips patients already present in the output file).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402
import run_human_drift_spine_000469 as spine469  # noqa: E402

SEED = 20260805
N_DRAWS = 200
RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "region_unit_count_matched_lambda.json"

PREDECLARED_DECISION = (
    "The regional confinement-rate ordering is supported only if, at matched unit counts and "
    "within patient, the pre-SMA-minus-hippocampus difference in lambda retains a patient-"
    "bootstrap interval excluding zero AND keeps the sign it has at unmatched counts. If the sign "
    "flips or the interval crosses zero at matched counts, the ordering is an identifiability "
    "artifact and is reported as such."
)
UNMATCHED_SIGN = "positive"  # pre_sma - hippocampus > 0 at unmatched counts (region_stratified_drift_000469.json)


def _stable_seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16) ^ SEED


def _eligible_patients() -> list[dict]:
    artifact = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())
    hipp = artifact["regions"]["hippocampus"]["sessions"]
    sma = artifact["regions"]["pre_sma"]["sessions"]
    out = []
    for patient in sorted(set(hipp) & set(sma)):
        h, s = hipp[patient], sma[patient]
        if h.get("status") == "complete" and s.get("status") == "complete":
            out.append({"patient": patient, "n_hippocampus": h["n_units"], "n_pre_sma": s["n_units"]})
    return out


def draw_and_fit(
    path: Path, region: str, n_available: int, target_count: int, base_seed: int, n_draws: int = N_DRAWS,
) -> list[dict]:
    """R draws without replacement of ``target_count`` units from the region's
    already region-filtered, firing-rate-QC'd pool of ``n_available`` units,
    refit with the existing per-fold machinery unchanged."""
    rng = np.random.default_rng(base_seed)
    draws = []
    for d in range(n_draws):
        indices = np.sort(rng.choice(n_available, size=target_count, replace=False))
        fit = spine469.analyze_session(path, seed=base_seed + d, region=region, unit_indices=indices)
        if fit["status"] == "complete":
            lam = fit["summary"]["state_space_lambda_identified_mean"]
            state_diffusions = [
                row["state_space"]["diffusion"] for row in fit["folds"]
                if row["state_space"]["status"] == "identifiable"
            ]
            dif = float(np.mean(state_diffusions)) if state_diffusions else None
            draws.append({
                "draw": d, "identified": lam is not None,
                "lambda_rate": lam, "diffusion": dif,
            })
        else:
            draws.append({
                "draw": d, "identified": False, "lambda_rate": None, "diffusion": None,
                "status": fit["status"],
            })
    return draws


def summarize_draws(draws: list[dict]) -> dict:
    identified = [d for d in draws if d["identified"] and d["lambda_rate"] is not None]
    fraction_identified = len(identified) / len(draws) if draws else 0.0
    if not identified:
        return {
            "status": "non_identified", "n_draws": len(draws), "n_identified_draws": 0,
            "fraction_identified": fraction_identified,
        }
    lambdas = np.array([d["lambda_rate"] for d in identified])
    diffusions = np.array([d["diffusion"] for d in identified if d["diffusion"] is not None])
    return {
        "status": "estimable", "n_draws": len(draws), "n_identified_draws": len(identified),
        "fraction_identified": fraction_identified,
        "median_lambda": float(np.median(lambdas)),
        "lambda_draw_percentile_2_5": float(np.percentile(lambdas, 2.5)),
        "lambda_draw_percentile_97_5": float(np.percentile(lambdas, 97.5)),
        "median_diffusion": float(np.median(diffusions)) if len(diffusions) else None,
    }


def _write(per_patient: dict, final: bool) -> None:
    matched_diffs = []
    for row in per_patient.values():
        d = row["within_patient_matched_draw_diff_pre_sma_minus_hippocampus"]["median"]
        if d is not None:
            matched_diffs.append(d)
    if len(matched_diffs) >= 2:
        rng = np.random.default_rng(SEED)
        _, lo, hi = bootstrap_ci(np.asarray(matched_diffs), np.mean, n_boot=5000, rng=rng)
        matched_group = {
            "status": "estimable", "n_patients": len(matched_diffs),
            "mean": float(np.mean(matched_diffs)), "median": float(np.median(matched_diffs)),
            "patient_bootstrap_ci95": [float(lo), float(hi)],
        }
        interval_excludes_zero = bool(lo > 0 or hi < 0)
        sign_at_matched = "positive" if matched_group["mean"] > 0 else "negative"
        ordering_verdict = (
            "supported"
            if interval_excludes_zero and sign_at_matched == UNMATCHED_SIGN
            else "identifiability_artifact"
        )
    else:
        matched_group = {
            "status": "non_identified", "n_patients": len(matched_diffs),
            "reason": f"fewer than 2 patients have a matched-count paired estimate ({len(matched_diffs)})",
        }
        interval_excludes_zero = False
        ordering_verdict = "non_identified"

    fraction_by_unit_count: dict[str, dict[str, float]] = {}
    for region_key in ("hippocampus", "pre_sma"):
        by_count: dict[int, list[float]] = {}
        for row in per_patient.values():
            summary = row[region_key]
            count = row["target_count"]
            by_count.setdefault(count, []).append(summary["fraction_identified"])
        fraction_by_unit_count[region_key] = {
            str(count): float(np.mean(fracs)) for count, fracs in sorted(by_count.items())
        }

    output = {
        "schema_version": "1.0.0", "analysis_id": "region_unit_count_matched_lambda",
        "trigger": "blocking unit-count-matched sensitivity for the pre-SMA-vs-hippocampus ordering",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "n_draws_per_patient_per_structure": N_DRAWS,
        "seed": SEED,
        "scope": (
            "DANDI 000469, hippocampus vs pre_sma only -- the deciding pair. "
            "Extending to every ordered structure pair in every eligible dataset "
            "is deferred; see the crack register."
        ),
        "predeclared_decision": PREDECLARED_DECISION,
        "unmatched_sign_pre_sma_minus_hippocampus": UNMATCHED_SIGN,
        "per_patient": per_patient,
        "run_complete": final,
        "n_patients_completed": len(per_patient),
        "matched_count_pre_sma_minus_hippocampus_lambda_diff": matched_group,
        "gate_decision": {
            "interval_excludes_zero": interval_excludes_zero,
            "sign_matches_unmatched": (
                matched_group.get("mean", 0) > 0 == (UNMATCHED_SIGN == "positive")
                if matched_group.get("status") == "estimable" else None
            ),
            "verdict": ordering_verdict,
        },
        "identification_fraction_by_matched_unit_count": fraction_by_unit_count,
        "hemisphere_stratified_matched_draws_sensitivity": {
            "status": "not_run",
            "reason": (
                "Deferred given the R>=200-draws-per-patient cost already incurred by the unit-count "
                "and identifiability matched-draws runs alone for one structure pair; hemisphere-"
                "stratified matched draws are a separate multiplicative dimension of the same expensive "
                "loop. Reported as an explicit deferral, not a silent omission."
            ),
        },
    }
    OUTPUT_PATH.write_text(canonical_json(output))


def main() -> None:
    patients = _eligible_patients()
    print(f"{len(patients)} patients eligible (both hippocampus and pre_sma 'complete' at unmatched counts)")
    directory = spine469.data_directory()
    per_patient: dict = {}
    if OUTPUT_PATH.exists():
        try:
            per_patient = json.loads(OUTPUT_PATH.read_text()).get("per_patient", {})
            if per_patient:
                print(f"resuming: {len(per_patient)} patients already completed")
        except Exception:
            per_patient = {}

    for row in patients:
        patient = row["patient"]
        if patient in per_patient:
            continue
        target_count = min(row["n_hippocampus"], row["n_pre_sma"])
        path = directory / patient / f"{patient}_ses-2_ecephys+image.nwb"
        print(f"{patient}: target_count={target_count} (hipp={row['n_hippocampus']}, sma={row['n_pre_sma']})",
              flush=True)
        hipp_draws = draw_and_fit(path, "hippocampus", row["n_hippocampus"], target_count,
                                   _stable_seed(patient, "hippocampus"))
        sma_draws = draw_and_fit(path, "pre_sma", row["n_pre_sma"], target_count,
                                  _stable_seed(patient, "pre_sma"))
        paired_diffs = [
            s["lambda_rate"] - h["lambda_rate"]
            for h, s in zip(hipp_draws, sma_draws)
            if h["identified"] and s["identified"]
        ]
        per_patient[patient] = {
            "target_count": target_count,
            "n_hippocampus_available": row["n_hippocampus"], "n_pre_sma_available": row["n_pre_sma"],
            "hippocampus": summarize_draws(hipp_draws), "pre_sma": summarize_draws(sma_draws),
            "within_patient_matched_draw_diff_pre_sma_minus_hippocampus": {
                "n_paired_draws_both_identified": len(paired_diffs),
                "median": float(np.median(paired_diffs)) if paired_diffs else None,
            },
        }
        _write(per_patient, final=False)
        print(f"  checkpointed ({len(per_patient)}/{len(patients)} patients done)", flush=True)

    _write(per_patient, final=True)
    print(json.dumps({"output": str(OUTPUT_PATH), "n_patients_completed": len(per_patient)}, indent=2))


if __name__ == "__main__":
    main()
