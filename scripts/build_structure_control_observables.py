#!/usr/bin/env python3
"""Per-structure control observables.

lambda and D are already fit per session per structure
(results/region_stratified_drift_000469.json,
results/region_stratified_drift_001187_000673.json's content_axis_battery
block). This script aggregates D under the same per-fold identifiability
filter lambda already uses -- D has never been aggregated at the structure
level before -- and derives three closed-form observables from the pair. No
new estimator, no controller, no LQR arm, no simulation, no digital twin
(6.4).

Run (after region_stratified_drift_000469.json and
region_stratified_drift_001187_000673.json's content_axis_battery both exist):
    conda run -n wm_dynamics python scripts/build_structure_control_observables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import ANATOMICAL_REGIONS  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402
from statistics import extract_000469_pairs, extract_001187_000673_pairs  # noqa: E402

RESULTS = ROOT / "results"
# State units: one unit along the leading FrozenPSTHTransform+PCA axis fit
# within each outer-training fold (see units_note below for why this is not
# comparable in absolute terms across structures/datasets).
DELTA_GRID_STATE_UNITS = [0.5, 1.0, 2.0, 4.0]


def bootstrap_summary(values: list[float], rng: np.random.Generator) -> dict[str, Any]:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {
            "status": "non_identified", "n_sessions": int(len(finite)),
            "reason": f"fewer than 2 finite estimates ({len(finite)})",
        }
    _, lo, hi = bootstrap_ci(finite, np.mean, n_boot=5000, rng=rng)
    return {
        "status": "estimable", "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "bootstrap_ci95": [float(lo), float(hi)], "n_sessions": int(len(finite)),
    }


def displacement_snr_grid(stationary_variance: float, delta_grid: list[float]) -> dict[str, float]:
    return {str(delta): float(delta / np.sqrt(stationary_variance)) for delta in delta_grid}


def build_structure_control_observables(
    pairs: list[tuple[float, float]], rng: np.random.Generator,
) -> dict[str, Any]:
    n = len(pairs)
    lambda_summary = bootstrap_summary([p[0] for p in pairs], rng)
    diffusion_summary = bootstrap_summary([p[1] for p in pairs], rng)
    bandwidth_ms = [1000.0 / lam for lam, _ in pairs if lam > 0]
    bandwidth_summary = bootstrap_summary(bandwidth_ms, rng)
    stationary_variance_values = [dif / lam for lam, dif in pairs if lam > 0]
    stationary_variance_summary = bootstrap_summary(stationary_variance_values, rng)
    if stationary_variance_summary["status"] == "estimable" and stationary_variance_summary["mean"] > 0:
        displacement_snr = displacement_snr_grid(stationary_variance_summary["mean"], DELTA_GRID_STATE_UNITS)
    else:
        displacement_snr = {"status": "non_identified", "reason": "stationary variance not identifiable"}
    return {
        "status": "estimable" if n >= 2 else "non_identified",
        "n_sessions_both_lambda_and_diffusion_identifiable": n,
        "control_bandwidth_ms": bandwidth_summary,
        "stationary_variance_state_units_sq": stationary_variance_summary,
        "displacement_snr_by_delta": displacement_snr,
        "lambda_s_per_s": lambda_summary,
        "diffusion_state_units_sq_per_s": diffusion_summary,
    }


UNITS_NOTE = (
    "lambda in s^-1 (confinement rate); D in (state units)^2/s where the state unit is one unit "
    "along the leading FrozenPSTHTransform+PCA axis fit within each outer-training fold "
    "(variance-normalized firing rate, z-scored per unit before PCA -- src/spike_pipeline.py "
    "FrozenPSTHTransform). This basis is refit independently per fold and per session, so it is NOT "
    "numerically comparable in absolute terms across structures, sessions, or datasets. "
    "displacement_snr_by_delta is therefore reported in this within-fit-standardized basis only, "
    "with this limitation stated explicitly rather than silently treated as physical units "
    "."
)
IDENTIFIABILITY_FILTER_NOTE = (
    "Session-level lambda and diffusion are each the mean over folds where "
    "state_space.status == 'identifiable' -- the SAME per-fold identifiability flag governs both, "
    "since one Gaussian state-space fit yields lambda and diffusion together, never separately. A "
    "session contributes to a structure's control observables only when this per-fold mean exists "
    "for both."
)


def main() -> None:
    path_469 = RESULTS / "region_stratified_drift_000469.json"
    path_1187 = RESULTS / "region_stratified_drift_001187_000673.json"
    missing = [p for p in (path_469, path_1187) if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required upstream artifacts: {[str(p) for p in missing]}")
    art_469 = json.loads(path_469.read_text())
    art_1187 = json.loads(path_1187.read_text())
    if "content_axis_battery" not in art_1187:
        raise SystemExit(
            f"{path_1187} has no content_axis_battery block yet -- run "
            "scripts/run_human_drift_spine_001187_000673.py --content-axis-region-stratified first "
            "."
        )

    rng = np.random.default_rng(20260804)
    structures = {}
    for region in ANATOMICAL_REGIONS:
        pairs = extract_000469_pairs(art_469, region) + extract_001187_000673_pairs(art_1187, region)
        structures[region] = build_structure_control_observables(pairs, rng)

    output = {
        "schema_version": "1.0.0", "analysis_id": "structure_control_observables",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "delta_grid_state_units": DELTA_GRID_STATE_UNITS,
        "units_note": UNITS_NOTE,
        "identifiability_filter": IDENTIFIABILITY_FILTER_NOTE,
        "source_artifacts": {
            "dandi_000469": str(path_469.relative_to(ROOT)),
            "dandi_001187_000673_content_axis_battery": str(path_1187.relative_to(ROOT)),
        },
        "note": (
            "Three observables computed from already-fitted lambda/D, no new "
            "controller/LQR/simulation/digital-twin. Scoped to the same two datasets/five "
            "structures results/structure_pooled_dynamics.json already covers -- DANDI 000574 "
            "and the LFP corpora are not in this artifact; folding them in is a full-corpus "
            "registry rebuild, not done here."
        ),
        "structures": structures,
    }
    (RESULTS / "structure_control_observables.json").write_text(canonical_json(output))
    print(json.dumps({r: structures[r]["status"] for r in structures}, indent=2))


if __name__ == "__main__":
    main()
