#!/usr/bin/env python3
"""run_persistence_patient_clustered_replication.py -- patient-clustered
companions to four paired/one-sample tests that were computed over sessions
treated as independent even though every session nests inside a patient.

results/band_versus_sensor_decomposition.json's depth-low-band positive
test, its paired depth-minus-scalp contrast, and its band-versus-band paired
test, plus results/observability_matched_modality_test.json's grain
discordance test, are each session-level: DANDI 000574 contributes roughly
four sessions per patient, so a session-level n overstates the number of
independent units. This script adds a patient-clustered version beside each
session-level one, computed from per-session values already on disk (read
from the checkpoints the producing scripts already wrote) -- no session is
refit.

Patient reduction, continuous quantities (depth level, scalp level, band
levels): each patient's sessions are reduced to that patient's median before
the same one-sample (t-test + Wilcoxon against zero) or paired (two-sided
sign-flip) test already used at the session level is re-run over patients.

Patient reduction, the grain-discordance binomial test: the session-level
test already restricts to discordant sessions (cross_validated_nugget_fraction
fits at exactly one of the two grains) because concordant sessions carry no
directional information by construction. The patient-clustered version keeps
that restriction, then reduces each patient's discordant sessions to that
patient's median signed code (+1 favours LFP, -1 favours the unit grain);
a patient whose discordant sessions split evenly has a tied (zero) median and
is excluded from the binomial test for the same reason a concordant session
is -- it carries no direction.

Deliverable: two new top-level fields, one per artifact, added extend-only.
Every pre-existing key in both artifacts is verified byte-identical before
and after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import locked_json_update  # noqa: E402
from provenance import canonical_json  # noqa: E402
from statistics import minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402
from run_band_versus_sensor_decomposition import load_existing_high_gamma_sessions  # noqa: E402
from run_band_versus_sensor_decomposition_extensions import (  # noqa: E402
    load_checkpoint_sessions, sessions_for_cell, _extract_persistence_level_factor_analysis,
)

BAND_SENSOR_ARTIFACT = ROOT / "results" / "band_versus_sensor_decomposition.json"
MATCHED_MODALITY_ARTIFACT = ROOT / "results" / "observability_matched_modality_test.json"
SEED = 20260813  # same date-stable convention as the extensions script this reuses


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


def _patient_median(values_by_patient_session: dict[tuple[str, str], float]) -> dict[str, float]:
    by_patient: dict[str, list[float]] = {}
    for (patient, _session), value in values_by_patient_session.items():
        by_patient.setdefault(patient, []).append(value)
    return {patient: float(np.median(vals)) for patient, vals in by_patient.items()}


def _one_sample_patient_stats(patient_values: dict[str, float]) -> dict:
    values = np.array(list(patient_values.values()), dtype=float)
    n = len(values)
    if n < 2:
        return {"status": "not_computable", "n_patients": n, "reason": "fewer than 2 patients with a fitted median"}
    t = stats.ttest_1samp(values, 0.0)
    w = stats.wilcoxon(values) if n >= 1 and np.any(values != 0) else None
    return {
        "status": "computed",
        "n_patients": n,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "n_positive": int((values > 0).sum()),
        "t_test_p_value": float(t.pvalue),
        "wilcoxon_p_value": float(w.pvalue) if w is not None else None,
        "per_patient_median": dict(sorted(patient_values.items())),
    }


def _paired_patient_stats(interest: dict[str, float], reference: dict[str, float], seed_parts: tuple) -> dict:
    shared = sorted(set(interest) & set(reference))
    n = len(shared)
    if n < 2:
        return {"status": "not_computable", "n_patients": n, "reason": "fewer than 2 patients with both arms fitted"}
    interest_arr = np.array([interest[p] for p in shared], dtype=float)
    reference_arr = np.array([reference[p] for p in shared], dtype=float)
    diffs = interest_arr - reference_arr
    test = paired_sign_flip_test(interest_arr, reference_arr, alternative="two-sided", rng=_seed(*seed_parts))
    mdd = minimum_detectable_paired_difference(diffs)
    return {
        "status": "computed",
        "n_patients": n,
        "patients": shared,
        "mean_difference_interest_minus_reference": float(diffs.mean()),
        "median_difference_interest_minus_reference": float(np.median(diffs)),
        "n_positive": int((diffs > 0).sum()),
        "p_value": test["p_value"],
        "ci_lower_mean_difference": test["ci_lower"],
        "ci_upper_mean_difference": test["ci_upper"],
        "minimum_detectable_paired_difference_80pct_power": mdd,
    }


def band_sensor_patient_clustered() -> dict:
    checkpoint_sessions = load_checkpoint_sessions()
    hg_by_bin: dict[int, dict] = {}
    for (patient, session, bin_ms), s in load_existing_high_gamma_sessions().items():
        hg_by_bin.setdefault(bin_ms, {})[(patient, session)] = s

    by_bin = {}
    for bin_ms in (100, 200):
        depth_b = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_b_bin{bin_ms}", bin_ms)
        scalp = sessions_for_cell(checkpoint_sessions, "scalp_low_band_bin{bin_ms}", bin_ms)
        depth_a = sessions_for_cell(checkpoint_sessions, "depth_low_band_comparison_a_bin{bin_ms}", bin_ms)
        hg = hg_by_bin.get(bin_ms, {})

        depth_b_levels = {k: v for k, v in ((key, _extract_persistence_level_factor_analysis(cell))
                                             for key, cell in depth_b.items()) if v is not None}
        scalp_levels = {k: v for k, v in ((key, _extract_persistence_level_factor_analysis(cell))
                                           for key, cell in scalp.items()) if v is not None}
        depth_a_levels = {k: v for k, v in ((key, _extract_persistence_level_factor_analysis(cell))
                                             for key, cell in depth_a.items()) if v is not None}
        hg_levels = {k: v for k, v in ((key, _extract_persistence_level_factor_analysis(cell))
                                        for key, cell in hg.items()) if v is not None}

        # Comparison_b's own pairing (depth vs scalp) is restricted to sessions present in both arms,
        # matching band_versus_sensor_persistence_reproduction's own construction.
        shared_b = set(depth_b_levels) & set(scalp_levels)
        depth_b_shared = {k: v for k, v in depth_b_levels.items() if k in shared_b}
        scalp_shared = {k: v for k, v in scalp_levels.items() if k in shared_b}

        depth_patient = _patient_median(depth_b_shared)
        scalp_patient = _patient_median(scalp_shared)
        depth_a_patient = _patient_median(depth_a_levels)
        hg_patient = _patient_median(hg_levels)

        by_bin[f"bin{bin_ms}"] = {
            "n_sessions_depth_scalp_pairing": len(shared_b),
            "n_sessions_band_pairing": len(set(depth_a_levels) & set(hg_levels)),
            "depth_low_band_one_sample": _one_sample_patient_stats(depth_patient),
            "scalp_low_band_one_sample": _one_sample_patient_stats(scalp_patient),
            "paired_depth_minus_scalp": _paired_patient_stats(
                depth_patient, scalp_patient, ("patient_depth_minus_scalp", bin_ms)),
            "band_versus_band_depth_low_minus_high_gamma": _paired_patient_stats(
                depth_a_patient, hg_patient, ("patient_band_versus_band", bin_ms)),
        }
    return {
        "method": (
            "each patient reduced to the median of that patient's session-level persistence-contrast "
            "levels (factor-analysis basis) before the same one-sample (t-test + Wilcoxon against zero) "
            "or two-sided paired sign-flip test already used at the session level "
            "(results/band_versus_sensor_decomposition.json's band_versus_sensor_persistence_reproduction "
            "and band_comparison_persistence_paired_test) is re-run over patients instead of sessions; "
            "session-level values and tests are unchanged and stand beside this as the descriptive number"
        ),
        "by_bin_width": by_bin,
    }


def grain_discordance_patient_clustered() -> dict:
    artifact = json.loads(MATCHED_MODALITY_ARTIFACT.read_text())
    mm = artifact["matched_modality_test_by_bin_width"]
    by_bin = {}
    for bin_key, test in mm.items():
        by_patient: dict[str, list[int]] = {}
        for row in test["sessions"]:
            if row["lfp_status"] == "fitted" and row["unit_status"] != "fitted":
                code = 1
            elif row["unit_status"] == "fitted" and row["lfp_status"] != "fitted":
                code = -1
            else:
                continue  # concordant at both or neither -- no directional information, excluded
            by_patient.setdefault(row["patient"], []).append(code)

        per_patient_median = {p: float(np.median(codes)) for p, codes in by_patient.items()}
        informative = {p: m for p, m in per_patient_median.items() if m != 0}
        n_favor_lfp = sum(1 for m in informative.values() if m > 0)
        n_favor_unit = sum(1 for m in informative.values() if m < 0)
        n_informative = len(informative)
        n_tied_excluded = len(per_patient_median) - n_informative

        if n_informative < 1:
            result = {
                "status": "not_computable",
                "n_patients_with_a_discordant_session": len(per_patient_median),
                "n_patients_tied_and_excluded": n_tied_excluded,
                "reason": "no patient has a non-tied discordant-session median at this bin width",
            }
        else:
            p_value = float(binomtest(n_favor_lfp, n_informative, 0.5, alternative="two-sided").pvalue)
            result = {
                "status": "computed",
                "n_patients_with_a_discordant_session": len(per_patient_median),
                "n_patients_tied_and_excluded": n_tied_excluded,
                "n_informative_patients": n_informative,
                "n_patients_favor_lfp": n_favor_lfp,
                "n_patients_favor_unit": n_favor_unit,
                "two_sided_exact_binomial_p_value": p_value,
                "per_patient_discordant_session_codes": dict(sorted(by_patient.items())),
            }
        by_bin[bin_key] = result
    return {
        "method": (
            "restricted to discordant sessions exactly as the session-level grain_discordance_paired_test "
            "already is (concordant-at-both-or-neither sessions carry no directional information by "
            "construction); each patient's remaining discordant sessions are then reduced to that "
            "patient's median signed code (+1 favours LFP, -1 favours the unit grain), a patient whose "
            "discordant sessions split evenly is tied (median 0) and excluded for the same reason a "
            "concordant session is, and the same exact two-sided binomial test against a 50/50 null is "
            "run over the remaining informative patients instead of sessions"
        ),
        "by_bin_width": by_bin,
    }


def main() -> None:
    band_sensor_new = {"patient_clustered_persistence_replication": band_sensor_patient_clustered()}
    with locked_json_update(BAND_SENSOR_ARTIFACT) as data:
        before = canonical_json({k: v for k, v in data.items() if k not in band_sensor_new})
        already_present = [f for f in band_sensor_new if f in data]
        if already_present:
            raise RuntimeError(f"refusing to overwrite already-present extend-only fields: {already_present}")
        data.update(band_sensor_new)
        after = canonical_json({k: v for k, v in data.items() if k not in band_sensor_new})
        if before != after:
            raise RuntimeError("extend-only violation: an existing band_versus_sensor_decomposition.json key would have changed")

    matched_modality_new = {"grain_discordance_patient_clustered_test": grain_discordance_patient_clustered()}
    with locked_json_update(MATCHED_MODALITY_ARTIFACT) as data:
        before = canonical_json({k: v for k, v in data.items() if k not in matched_modality_new})
        already_present = [f for f in matched_modality_new if f in data]
        if already_present:
            raise RuntimeError(f"refusing to overwrite already-present extend-only fields: {already_present}")
        data.update(matched_modality_new)
        after = canonical_json({k: v for k, v in data.items() if k not in matched_modality_new})
        if before != after:
            raise RuntimeError("extend-only violation: an existing observability_matched_modality_test.json key would have changed")

    print(f"extended {BAND_SENSOR_ARTIFACT.name} with patient_clustered_persistence_replication", file=sys.stderr)
    print(f"extended {MATCHED_MODALITY_ARTIFACT.name} with grain_discordance_patient_clustered_test", file=sys.stderr)
    bs = band_sensor_new["patient_clustered_persistence_replication"]["by_bin_width"]
    mm = matched_modality_new["grain_discordance_patient_clustered_test"]["by_bin_width"]
    print(json.dumps({
        "bin100_depth_n_patients": bs["bin100"]["depth_low_band_one_sample"].get("n_patients"),
        "bin100_paired_depth_minus_scalp": bs["bin100"]["paired_depth_minus_scalp"],
        "bin200_paired_depth_minus_scalp": bs["bin200"]["paired_depth_minus_scalp"],
        "bin100_band_versus_band": bs["bin100"]["band_versus_band_depth_low_minus_high_gamma"],
        "bin200_band_versus_band": bs["bin200"]["band_versus_band_depth_low_minus_high_gamma"],
        "grain_discordance_bin100": mm["bin100"],
        "grain_discordance_bin200": mm["bin200"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
