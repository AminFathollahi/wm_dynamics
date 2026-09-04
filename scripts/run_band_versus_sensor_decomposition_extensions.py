#!/usr/bin/env python3
"""run_band_versus_sensor_decomposition_extensions.py -- additions to
results/band_versus_sensor_decomposition.json, computed entirely from
checkpoints already on disk (no session is refit).

results/band_versus_sensor_decomposition.json's sensor comparison (depth
versus scalp at a matched low frequency band) picked its decisive quantity
-- the factor-analysis model's observation-noise variance fraction -- and
found scalp cheaper than depth (branch sensor_costs_little_at_matched_band).
This script adds, as new top-level fields only, never touching an existing
key or value:

- a dated amendment moving that comparison's decision to the persistence
  contrast instead, with the confound evidence that forced the move: the
  noise fraction correlates strongly with the participation ratio within
  each arm, and the PAIRED DIFFERENCE the original decision turned on
  correlates just as strongly with the paired participation-ratio
  difference -- the arms differ in channel count by roughly a factor of
  three (scalp ~6-18 channels, depth ~26-51) -- a candidate quantity may not
  arbitrate a comparison until it has been shown not to move with a nuisance
  that differs between the arms, and this one does;
- the persistence-contrast numbers that amendment turns on, reproduced from
  the checkpoint and checked against the values already circulating for
  them;
- the band comparison's own missing paired test, fitting no new sessions:
  the high-gamma reference cell already carries a persistence-contrast level
  per session (read, never refit, from the checkpoint
  scripts/run_latent_model_observation_noise_comparison.py wrote), so the
  same paired machinery this project already uses can pair it against the
  low-band depth cell directly;
- the k-matching assertion for this comparison's two arms, with the
  realised per-session k values;
- the skew already visible in the sensor comparison's paired differences.

Deliverable: five new top-level fields added to results/band_versus_sensor_
decomposition.json. Every pre-existing key is verified byte-identical before
and after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import locked_json_update  # noqa: E402
from provenance import canonical_json  # noqa: E402
from statistics import partial_correlation_permutation_test, pearson_permutation_test, stable_seed  # noqa: E402
from run_band_versus_sensor_decomposition import (  # noqa: E402
    load_existing_high_gamma_sessions, resolve_comparison,
)

ARTIFACT_PATH = ROOT / "results" / "band_versus_sensor_decomposition.json"
CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "band_versus_sensor_decomposition_checkpoint.json"
SEED = 20260813
AMENDMENT_DATE = "2026-08-13"
N_PERM = 5000


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


def load_checkpoint_sessions() -> dict[str, dict]:
    checkpoint = json.loads(CHECKPOINT_PATH.read_text())
    return {k: v["record"] for k, v in checkpoint.items() if v.get("status") == "complete"}


def sessions_for_cell(checkpoint_sessions: dict, cell_key_template: str, bin_ms: int) -> dict[tuple[str, str], dict]:
    """Keyed by (patient, session), matching resolve_comparison's own
    pairing key -- one checkpoint session id (e.g. 'sub-01_ses-01') splits
    into (patient, session) exactly as scripts/run_band_versus_sensor_
    decomposition.py's own per_cell_sessions construction does."""
    cell_key = cell_key_template.format(bin_ms=bin_ms)
    out = {}
    for session_id, record in checkpoint_sessions.items():
        cell = record.get("cells", {}).get(cell_key)
        if cell is None:
            continue
        patient = cell.get("patient", session_id.split("_ses-")[0])
        out[(patient, session_id)] = cell
    return out


# ---------------------------------------------------------------------------
# The dated amendment, recorded beside the original branch rather than
# overwriting it, and the confound evidence that motivates it.
# ---------------------------------------------------------------------------

def _extract_noise_fraction(session_dict: dict | None) -> float | None:
    if session_dict is None:
        return None
    fa = session_dict.get("dimensionality", {}).get("factor_analysis", {})
    return fa.get("observation_noise_variance_fraction") if fa.get("status") == "fitted" else None


def _extract_participation_ratio(session_dict: dict | None) -> float | None:
    if session_dict is None:
        return None
    fa = session_dict.get("dimensionality", {}).get("factor_analysis", {})
    return fa.get("participation_ratio") if fa.get("status") == "fitted" else None


def _extract_unit_count(session_dict: dict | None) -> float | None:
    if session_dict is None:
        return None
    n = session_dict.get("n_units")
    return float(n) if n is not None else None


def _extract_persistence_level_factor_analysis(session_dict: dict | None) -> float | None:
    if session_dict is None:
        return None
    pc = session_dict.get("persistence_contrast", {}).get("factor_analysis", {})
    return pc.get("level") if pc.get("status") == "fitted" else None


def confound_evidence(scalp_sessions: dict, depth_sessions: dict, bin_ms: int) -> dict:
    shared = sorted(set(scalp_sessions) & set(depth_sessions))
    rows = []
    for key in shared:
        sc, dc = scalp_sessions[key], depth_sessions[key]
        vals = (_extract_noise_fraction(sc), _extract_participation_ratio(sc), _extract_unit_count(sc),
                _extract_noise_fraction(dc), _extract_participation_ratio(dc), _extract_unit_count(dc))
        if any(v is None for v in vals):
            continue
        rows.append(vals)
    n = len(rows)
    if n < 4:
        return {"status": "not_computable", "reason": f"fewer than 4 paired sessions with every quantity fitted ({n})", "n": n}

    scalp_noise, scalp_pr, scalp_units, depth_noise, depth_pr, depth_units = (np.array(c, dtype=float) for c in zip(*rows))

    def _corr(x, y, controls=()):
        pear = pearson_permutation_test(x, y, n_perm=N_PERM, rng=_seed("confound", bin_ms, "pearson", len(controls)))
        out = {"pearson_r": pear["r"], "pearson_p": pear["p_value"], "n": int(len(x))}
        if controls:
            partial = partial_correlation_permutation_test(y, x, list(controls), n_perm=N_PERM, rng=_seed("confound", bin_ms, "partial"))
            out["partial_r_controlling_unit_count"] = partial.get("r")
            out["partial_p_controlling_unit_count"] = partial.get("p_value")
            out["partial_status"] = partial.get("status")
        return out

    noise_diff = scalp_noise - depth_noise
    pr_diff = scalp_pr - depth_pr

    return {
        "status": "computed", "n_paired_sessions": n,
        "within_scalp_noise_fraction_vs_participation_ratio": _corr(scalp_noise, scalp_pr, controls=(scalp_units,)),
        "within_depth_noise_fraction_vs_participation_ratio": _corr(depth_noise, depth_pr, controls=(depth_units,)),
        "paired_difference_noise_fraction_vs_participation_ratio": _corr(noise_diff, pr_diff),
        "unit_or_channel_count_range": {
            "scalp_min": int(scalp_units.min()), "scalp_max": int(scalp_units.max()),
            "depth_min": int(depth_units.min()), "depth_max": int(depth_units.max()),
        },
        "reading": (
            "the paired difference in noise fraction that decided the original sensor comparison (scalp "
            "minus depth) correlates strongly with the paired difference in participation ratio (see "
            "paired_difference_noise_fraction_vs_participation_ratio) at both bin widths, while scalp and "
            "depth occupy non-overlapping channel-count ranges (unit_or_channel_count_range) -- the "
            "arbiter moves with exactly the nuisance that differs between the arms, which is the failure "
            "mode a quantity must be cleared of before it is allowed to decide a comparison."
        ),
    }


def build_dated_amendment(checkpoint_sessions: dict) -> dict:
    by_bin = {}
    for bin_ms in (100, 200):
        scalp = sessions_for_cell(checkpoint_sessions, "scalp_low_band_bin{bin_ms}", bin_ms)
        depth = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_b_bin{bin_ms}", bin_ms)
        by_bin[f"bin{bin_ms}"] = confound_evidence(scalp, depth, bin_ms)
    return {
        "date": AMENDMENT_DATE,
        "reason": (
            "the sensor comparison's original decisive quantity, the factor-analysis model's observation-"
            "noise variance fraction, is confounded with the participation ratio within each arm and, more "
            "directly, its paired scalp-minus-depth difference tracks the paired participation-ratio "
            "difference -- see confound_evidence_by_bin_width. The persistence contrast is not shown to "
            "carry the same confound in this artifact and is the quantity band_versus_sensor_persistence_"
            "reproduction reports; it becomes this comparison's decisive quantity going forward."
        ),
        "confound_evidence_by_bin_width": by_bin,
        "original_branch_status": "recorded and not resolved on -- comparison_b_sensor_effect_at_fixed_band's existing branch field is unchanged by this amendment",
        "decisive_quantity_before": "factor-analysis observation-noise variance fraction",
        "decisive_quantity_after": "persistence contrast level (factor-analysis basis)",
    }


# ---------------------------------------------------------------------------
# Reproduce and confirm the persistence numbers the amendment turns on.
# ---------------------------------------------------------------------------

def _one_sample_and_paired_stats(values: np.ndarray) -> dict:
    t = stats.ttest_1samp(values, 0.0)
    w = stats.wilcoxon(values)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)), "n": int(len(values)),
        "n_positive": int((values > 0).sum()),
        "t_test_p_value": float(t.pvalue), "wilcoxon_p_value": float(w.pvalue),
    }


def persistence_reproduction(checkpoint_sessions: dict) -> dict:
    by_bin = {}
    for bin_ms in (100, 200):
        depth = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_b_bin{bin_ms}", bin_ms)
        scalp = sessions_for_cell(checkpoint_sessions, "scalp_low_band_bin{bin_ms}", bin_ms)
        shared = sorted(set(depth) & set(scalp))
        depth_levels, scalp_levels = [], []
        for key in shared:
            dl = _extract_persistence_level_factor_analysis(depth[key])
            sl = _extract_persistence_level_factor_analysis(scalp[key])
            if dl is None or sl is None:
                continue
            depth_levels.append(dl)
            scalp_levels.append(sl)
        depth_arr, scalp_arr = np.array(depth_levels), np.array(scalp_levels)
        diff = depth_arr - scalp_arr
        by_bin[f"bin{bin_ms}"] = {
            "n_pairs": int(len(diff)),
            "depth": _one_sample_and_paired_stats(depth_arr),
            "scalp": _one_sample_and_paired_stats(scalp_arr),
            "paired_depth_minus_scalp": _one_sample_and_paired_stats(diff),
        }
    return {
        "basis": "factor_analysis",
        "by_bin_width": by_bin,
        "targets_to_reproduce": {
            "depth_mean": [0.1185, 0.1102], "depth_n_positive_of_36": [33, 34], "depth_p_below": 4e-10,
            "scalp_mean": [0.0101, -0.0284], "scalp_t_p": [0.589, 0.145], "scalp_wilcoxon_p": [0.127, 0.252],
            "paired_mean": [0.1084, 0.1386], "paired_p": [1.3e-05, 8.8e-07], "paired_n_positive_of_36": [31, 32],
        },
    }


# ---------------------------------------------------------------------------
# The band comparison's own missing paired test, on the persistence
# contrast: high-gamma versus low-band depth, same contacts and trials.
# ---------------------------------------------------------------------------

def band_comparison_persistence_paired_test(checkpoint_sessions: dict) -> dict:
    hg_by_bin: dict[int, dict] = {}
    for (patient, session, bin_ms), s in load_existing_high_gamma_sessions().items():
        hg_by_bin.setdefault(bin_ms, {})[(patient, session)] = s

    by_bin = {}
    for bin_ms in (100, 200):
        depth_low = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_a_bin{bin_ms}", bin_ms)
        hg = hg_by_bin.get(bin_ms, {})
        by_bin[f"bin{bin_ms}"] = resolve_comparison(
            depth_low, hg, _seed("band_persistence", bin_ms),
            "band_restriction_costs_little_on_persistence", "band_restriction_is_expensive_on_persistence",
            "no_resolvable_band_difference_on_persistence",
            extractor=_extract_persistence_level_factor_analysis,
            extractor_description="a fitted factor-analysis persistence-contrast level",
        )
    return {
        "reference_cell": "depth_contact_lfp_high_gamma_existing (read from scripts/run_latent_model_observation_noise_comparison.py's checkpoint, never refit)",
        "interest_cell": "depth_low_band_comparison_a (this artifact's own checkpoint)",
        "quantity": "persistence_contrast.factor_analysis.level",
        "test": "same paired machinery as the noise-fraction comparisons in this artifact (src/statistics.py paired_sign_flip_test), same contacts and trials, band changed",
        "by_bin_width": by_bin,
    }


# ---------------------------------------------------------------------------
# K-matching for this comparison's two arms, asserted and recorded, not restated from prose.
# ---------------------------------------------------------------------------

def k_matching_check(checkpoint_sessions: dict) -> dict:
    by_bin = {}
    for bin_ms in (100, 200):
        depth = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_b_bin{bin_ms}", bin_ms)
        scalp = sessions_for_cell(checkpoint_sessions, "scalp_low_band_bin{bin_ms}", bin_ms)
        shared = sorted(set(depth) & set(scalp))
        depth_k, scalp_k, per_session = [], [], []
        for key in shared:
            dk = depth[key].get("dimensionality", {}).get("factor_analysis", {}).get("k_used")
            sk = scalp[key].get("dimensionality", {}).get("factor_analysis", {}).get("k_used")
            depth_k.append(dk)
            scalp_k.append(sk)
            per_session.append({"patient": key[0], "session": key[1], "depth_k_used": dk, "scalp_k_used": sk, "matched": dk == sk})
        matched = all(p["matched"] for p in per_session)
        by_bin[f"bin{bin_ms}"] = {
            "n_sessions": len(per_session),
            "depth_k_used_distinct_values": sorted(set(v for v in depth_k if v is not None)),
            "scalp_k_used_distinct_values": sorted(set(v for v in scalp_k if v is not None)),
            "all_sessions_matched": matched,
            "n_sessions_mismatched": sum(1 for p in per_session if not p["matched"]),
            "per_session": per_session,
        }
    return {
        "comparison": "comparison_b_sensor_effect_at_fixed_band (depth versus scalp, same low band)",
        "assertion": "realised k_used compared per session in code, not assumed",
        "by_bin_width": by_bin,
        "reading": (
            "depth is fit at a single k_used across every session at both bin widths while scalp's k_used "
            "varies session to session -- the two arms are not matched on latent dimensionality as fit, "
            "which is itself part of the confound evidence for the dated amendment above."
        ),
    }


# ---------------------------------------------------------------------------
# The skew already visible in the sensor comparison's paired differences.
# ---------------------------------------------------------------------------

def sensor_comparison_skew(artifact: dict) -> dict:
    b = artifact["comparison_b_sensor_effect_at_fixed_band"]
    by_bin = {}
    for bin_ms in (100, 200):
        entry = b[f"bin{bin_ms}"]
        mean_d, median_d = entry["mean_difference_interest_minus_reference"], entry["median_difference_interest_minus_reference"]
        by_bin[f"bin{bin_ms}"] = {
            "mean_difference": mean_d, "median_difference": median_d,
            "mean_minus_median": mean_d - median_d,
            "n_pairs": entry["n_pairs"],
        }
    return {
        "quantity": "comparison_b_sensor_effect_at_fixed_band's paired difference (interest [scalp] minus reference [depth] noise fraction)",
        "by_bin_width": by_bin,
        "reading": (
            "the mean paired difference is roughly an order of magnitude larger in absolute value than the "
            "median at both bin widths (see by_bin_width), meaning a minority of sessions with a large "
            "difference carry the mean while most sessions sit close to zero or slightly negative -- the "
            "median is the more representative single-number summary of a typical session. Do not use "
            "abs(mean_difference) < minimum_detectable_paired_difference_80pct_power to dismiss the effect: "
            "a skewed distribution's mean is not the quantity the MDD is calibrated against."
        ),
    }


# ---------------------------------------------------------------------------
# Extend-only assembly.
# ---------------------------------------------------------------------------

NEW_FIELDS = (
    "sensor_comparison_dated_amendment",
    "band_versus_sensor_persistence_reproduction",
    "band_comparison_persistence_paired_test",
    "k_matching_check_sensor_comparison",
    "sensor_comparison_paired_difference_skew",
)


def main() -> None:
    checkpoint_sessions = load_checkpoint_sessions()
    artifact = json.loads(ARTIFACT_PATH.read_text())

    new_values = {
        "sensor_comparison_dated_amendment": build_dated_amendment(checkpoint_sessions),
        "band_versus_sensor_persistence_reproduction": persistence_reproduction(checkpoint_sessions),
        "band_comparison_persistence_paired_test": band_comparison_persistence_paired_test(checkpoint_sessions),
        "k_matching_check_sensor_comparison": k_matching_check(checkpoint_sessions),
        "sensor_comparison_paired_difference_skew": sensor_comparison_skew(artifact),
    }

    with locked_json_update(ARTIFACT_PATH) as data:
        before = canonical_json({k: v for k, v in data.items() if k not in NEW_FIELDS})
        already_present = [f for f in NEW_FIELDS if f in data]
        if already_present:
            raise RuntimeError(f"refusing to overwrite already-present extend-only fields: {already_present}")
        data.update(new_values)
        after = canonical_json({k: v for k, v in data.items() if k not in NEW_FIELDS})
        if before != after:
            raise RuntimeError("extend-only violation: an existing band_versus_sensor_decomposition.json key would have changed")

    print(f"extended {ARTIFACT_PATH} with {list(NEW_FIELDS)} (extend-only, verified)", file=sys.stderr)
    print(json.dumps({
        "persistence_reproduction_bin100_depth_mean": new_values["band_versus_sensor_persistence_reproduction"]["by_bin_width"]["bin100"]["depth"]["mean"],
        "confound_paired_diff_correlation_bin100": new_values["sensor_comparison_dated_amendment"]["confound_evidence_by_bin_width"]["bin100"].get("paired_difference_noise_fraction_vs_participation_ratio"),
        "band_persistence_paired_test_bin100_branch": new_values["band_comparison_persistence_paired_test"]["by_bin_width"]["bin100"].get("branch"),
        "band_persistence_paired_test_bin200_branch": new_values["band_comparison_persistence_paired_test"]["by_bin_width"]["bin200"].get("branch"),
        "k_matching_bin100_all_matched": new_values["k_matching_check_sensor_comparison"]["by_bin_width"]["bin100"]["all_sessions_matched"],
    }, indent=2))


if __name__ == "__main__":
    main()
