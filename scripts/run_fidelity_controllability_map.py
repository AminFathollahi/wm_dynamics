#!/usr/bin/env python3
"""The fidelity-versus-controllability map: does the structure that holds a
memory best (small stationary variance D/lambda) pay for that with a
shorter stimulation window (small 1/lambda)?

Both quantities are already fit jointly per session in
results/structure_control_observables.json, under the SAME per-fold
identifiability filter (one Gaussian state-space fit yields lambda and D
together, never separately) -- this script adds only the cross-structure
rank-correlation test, with a session-level bootstrap that resamples the
underlying (lambda, diffusion) session pairs (not the already-summarized
point estimates), reusing
build_structure_control_observables.extract_000469_pairs /
extract_001187_000673_pairs rather than re-deriving them.

The cross-structure displacement-SNR column
(structure_control_observables.json#displacement_snr_by_delta) is dropped
here: Delta is in per-structure, per-fold PCA latent units with no common
scale across structures, so it licenses no cross-structure comparison. Only
the scale-free ratio (D/lambda) and the ms-valued bandwidth (1/lambda) are
used -- both are on a common (time, or a ratio of matched state-space units)
scale across structures.

No controller, no LQR/LQG, no closed-loop simulation, no digital twin: this
is closed-form algebra on parameters that already exist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import ANATOMICAL_REGIONS  # noqa: E402
from build_structure_control_observables import (  # noqa: E402
    extract_000469_pairs, extract_001187_000673_pairs,
)

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "fidelity_controllability_map.json"
SEED = 20260807
N_BOOT = 5000
MIN_STRUCTURES_FOR_CORRELATION = 3

PREDECLARED_DECISION = {
    "tradeoff": (
        "the D/lambda ordering and the 1/lambda ordering agree (rank-correlation interval "
        "excludes zero and is positive, in the fidelity-versus-latency sense: structures that "
        "hold memory well also have a short intervention window). The paper states the exchange "
        "rate in ms of intervention window per unit of stationary variance."
    ),
    "dissociation": (
        "the rank-correlation interval excludes zero and is negative: the orderings differ, some "
        "structure is both well-held and steerable, and it is the target a prosthetic should "
        "pick -- named with its interval."
    ),
    "estimator_non_identified": (
        "fewer than 3 structures have both quantities estimable, OR the rank-correlation interval "
        "spans (near) the full [-1, 1] range, OR the interval includes zero without excluding it "
        "in either direction -- none of these can decide tradeoff vs. dissociation. Per-structure "
        "values with their intervals are reported and no ordering claim is made."
    ),
    "note": "neither tradeoff nor dissociation is preferred.",
}


def _pairs_by_region() -> dict[str, list[tuple[float, float]]]:
    art_469 = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())
    art_1187 = json.loads((RESULTS / "region_stratified_drift_001187_000673.json").read_text())
    return {
        region: extract_000469_pairs(art_469, region) + extract_001187_000673_pairs(art_1187, region)
        for region in ANATOMICAL_REGIONS
    }


def _structure_means(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    stationary_variance = float(np.mean([d / l for l, d in pairs if l > 0]))
    bandwidth_ms = float(np.mean([1000.0 / l for l, d in pairs if l > 0]))
    return stationary_variance, bandwidth_ms


def rank_correlation_with_bootstrap(pairs_by_region: dict[str, list[tuple[float, float]]],
                                     rng: np.random.Generator) -> dict:
    regions = sorted(r for r, pairs in pairs_by_region.items() if len(pairs) >= 2)
    if len(regions) < MIN_STRUCTURES_FOR_CORRELATION:
        return {
            "status": "non_identified", "n_structures": len(regions), "regions": regions,
            "reason": f"fewer than {MIN_STRUCTURES_FOR_CORRELATION} structures with >=2 identified sessions",
        }
    variances, bandwidths = zip(*(_structure_means(pairs_by_region[r]) for r in regions))
    observed_rho, _ = spearmanr(variances, bandwidths)

    boot_rhos = np.empty(N_BOOT)
    for b in range(N_BOOT):
        v_boot, w_boot = [], []
        for r in regions:
            pairs = pairs_by_region[r]
            resample_idx = rng.integers(0, len(pairs), len(pairs))
            resampled = [pairs[i] for i in resample_idx]
            v, w = _structure_means(resampled)
            v_boot.append(v)
            w_boot.append(w)
        rho, _ = spearmanr(v_boot, w_boot)
        boot_rhos[b] = rho if np.isfinite(rho) else 0.0
    lo, hi = float(np.percentile(boot_rhos, 2.5)), float(np.percentile(boot_rhos, 97.5))
    return {
        "status": "estimable", "n_structures": len(regions), "regions": regions,
        "observed_spearman_rho": float(observed_rho), "bootstrap_ci95": [lo, hi],
        "excludes_zero_positive": bool(lo > 0), "excludes_zero_negative": bool(hi < 0),
        "spans_near_full_range": bool(lo < -0.9 and hi > 0.9),
        "n_boot": N_BOOT,
    }


def three_branch_verdict(correlation: dict) -> dict:
    if correlation.get("status") != "estimable":
        return {"verdict": "estimator_non_identified", "reason": correlation.get("reason")}
    if correlation["spans_near_full_range"]:
        return {"verdict": "estimator_non_identified", "reason": "rank-correlation interval spans (near) the full [-1, 1] range"}
    if correlation["excludes_zero_positive"]:
        return {"verdict": "tradeoff", "reason": "rank-correlation interval excludes zero and is positive"}
    if correlation["excludes_zero_negative"]:
        return {"verdict": "dissociation", "reason": "rank-correlation interval excludes zero and is negative"}
    return {"verdict": "estimator_non_identified", "reason": "rank-correlation interval includes zero; cannot decide tradeoff vs. dissociation"}


def main() -> None:
    observables = json.loads((RESULTS / "structure_control_observables.json").read_text())
    rng = np.random.default_rng(SEED)
    pairs_by_region = _pairs_by_region()
    correlation = rank_correlation_with_bootstrap(pairs_by_region, rng)
    verdict = three_branch_verdict(correlation)

    per_structure = {}
    for region, block in observables["structures"].items():
        if block.get("status") != "estimable":
            per_structure[region] = {"status": block.get("status", "non_identified")}
            continue
        per_structure[region] = {
            "status": "estimable",
            "stationary_variance_D_over_lambda": block["stationary_variance_state_units_sq"],
            "control_bandwidth_ms_1_over_lambda": block["control_bandwidth_ms"],
            "n_sessions": block["n_sessions_both_lambda_and_diffusion_identifiable"],
        }

    fidelity_ordering = sorted(
        (r for r in per_structure if per_structure[r]["status"] == "estimable"),
        key=lambda r: per_structure[r]["stationary_variance_D_over_lambda"]["mean"],
    )
    controllability_ordering = sorted(
        (r for r in per_structure if per_structure[r]["status"] == "estimable"),
        key=lambda r: per_structure[r]["control_bandwidth_ms_1_over_lambda"]["mean"],
    )

    output = {
        "schema_version": "1.0.0", "analysis_id": "fidelity_controllability_map",
        "trigger": "whether structures that hold memory well are for that reason harder to steer",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "seed": SEED,
        "scope": (
            "Same two datasets/five structures as results/structure_control_observables.json "
            "(DANDI 000469, DANDI 001187/000673 content_axis_battery); 000574/Boran and the LFP "
            "corpora are not in the source artifact this builds on."
        ),
        "scale_declaration": (
            "Only scale-free quantities are compared across structures -- the ratio D/lambda "
            "(stationary variance, in each structure's own PCA-latent state-units-squared, not "
            "compared in absolute magnitude across structures, only its rank is) and the "
            "ms-valued control bandwidth 1/lambda (a common physical scale, time). The "
            "cross-structure displacement_snr_by_delta column in "
            "structure_control_observables.json is superseded: Delta has no common scale across "
            "structures. See PAPER_REPORT.tex for where this is struck."
        ),
        "primary_grain_declaration": (
            "Session grain is primary for this map (matches structure_control_observables.json, "
            "which this artifact is built from). Amygdala lambda at session grain (1.641) differs "
            "8.8% from the patient grain in structure_registry.json (1.509). No figure or "
            "sentence built from this artifact may mix the two grains."
        ),
        "per_structure": per_structure,
        "fidelity_ordering_best_held_first": fidelity_ordering,
        "controllability_ordering_shortest_window_first": controllability_ordering,
        "rank_correlation_stationary_variance_vs_bandwidth": correlation,
        "predeclared_decision": {**PREDECLARED_DECISION, **verdict},
    }
    OUTPUT_PATH.write_text(canonical_json(output))
    print(json.dumps({
        "output": str(OUTPUT_PATH), "verdict": verdict["verdict"],
        "fidelity_ordering": fidelity_ordering, "controllability_ordering": controllability_ordering,
        "rank_correlation": correlation.get("observed_spearman_rho"),
        "ci": correlation.get("bootstrap_ci95"),
    }, indent=2))


if __name__ == "__main__":
    main()
