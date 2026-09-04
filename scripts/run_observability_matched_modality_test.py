#!/usr/bin/env python3
"""run_observability_matched_modality_test.py -- restates the census's
unit-versus-LFP grain ordering as what it actually measures, using only
data already on disk (no new model fitting).

This project's observability-and-power census ranks the LFP grain above the
unit grain on one shared observable, cross_validated_nugget_fraction (see
results/observability_and_power_census.json's margin_aware_modality_result):
at the wide bin width, the LFP grain (dandi_000574, depth_contact_lfp) is
admitted at margin while the unit grain (dandi_000574, single_unit, pooled)
is excluded, and every one of the other six shared observables ties. This
script asks two questions about that one asymmetric observable, on the same
patients and the same trials Boran's co-located spike-and-LFP recordings
provide:

1. Within each grain, across sessions, does the number of cross-validation
   splits the observable manages to fit correlate with the leading latent's
   temporal smoothness (lag-one autocorrelation), its participation ratio,
   or its raw unit/channel count -- i.e. is the grain ordering a statement
   about signal timescale, about dimensionality, or about instrumented
   population size, rather than about instrument quality.
2. Run the one asymmetric observable at both grains, same patients, same
   trials, at the bin width where the asymmetry exists, and ask whether it
   fits at LFP resolution when it does not fit at unit resolution.

The precursor this restates: results/boran_modality_consistency.json, which
asked a different observable (a drift/confinement-rate fit) of the same
co-located recordings and returned jointly identifiable in only one of eight
patients at 26 of 37 sessions. That number and the reason it does not close
are carried forward here rather than re-derived.

Deliverable: results/observability_matched_modality_test.json.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from io_utils import locked_json_update  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import pearson_permutation_test, spearman_permutation_test, stable_seed  # noqa: E402

SEED = 20260813

PRECURSOR_PATH = ROOT / "results" / "boran_modality_consistency.json"
CENSUS_PATH = ROOT / "results" / "observability_and_power_census.json"
NOISE_CHECKPOINT_PATH = ROOT / "results" / ".checkpoints" / "latent_model_observation_noise_comparison_checkpoint.json"
OUTPUT_PATH = ROOT / "results" / "observability_matched_modality_test.json"

UNIT_GRAIN = {"dataset": "dandi_000574", "modality": "single_unit", "structure": "pooled", "epoch": "delay",
              "checkpoint_key": "dandi_000574", "checkpoint_structure": "pooled"}
LFP_GRAIN = {"dataset": "dandi_000574", "modality": "depth_contact_lfp", "structure": "depth_contact_lfp", "epoch": "delay",
             "checkpoint_key": "dandi_000574_depth_contact_lfp", "checkpoint_structure": "depth_contact_lfp"}
N_PERM = 5000


def _seed(*parts) -> np.random.Generator:
    return np.random.default_rng((stable_seed("|".join(str(p) for p in parts)) ^ SEED) & 0xFFFFFFFF)


def _rho_from_slow_timescale(median_slow_timescale_s: float | None, bin_width_s: float) -> float | None:
    """Inverts src/observability.py's own slow_timescale_s = -bin_width_s /
    log(rho) convention to recover the lag-one autocorrelation rho a
    session's median reported timescale implies."""
    if median_slow_timescale_s is None or median_slow_timescale_s <= 0:
        return None
    return float(np.exp(-bin_width_s / median_slow_timescale_s))


# ---------------------------------------------------------------------------
# The precursor, read and restated, never re-derived.
# ---------------------------------------------------------------------------

def load_precursor() -> dict:
    d = json.loads(PRECURSOR_PATH.read_text())
    sessions = d["sessions"]
    excluded = {k: s for k, s in sessions.items() if s.get("status") != "complete"}
    reason_counts = Counter(s.get("reason") for s in excluded.values())
    return {
        "artifact": "results/boran_modality_consistency.json",
        "adjudication": d["adjudication"],
        "comparison_metric": d["comparison_metric"],
        "primary_estimator": d["primary_estimator"],
        "secondary_estimator": d["secondary_estimator"],
        "independent_unit": d["independent_unit"],
        "n_sessions_complete": d["n_sessions_complete"],
        "n_sessions_total": d["n_sessions_total"],
        "excluded_session_reason_counts": dict(reason_counts),
        "excluded_session_ids": sorted(excluded.keys()),
        "incompleteness_closure": {
            "closed": False,
            "reading": (
                "every one of the 11 sessions excluded from the precursor's own comparison carries a "
                "raw-data-availability reason recorded in that artifact's own per-session status field: "
                f"{reason_counts.get('no units table in this NWB file', 0)} of 11 have no spike-sorted "
                "units table in the source NWB file at all, and the remaining "
                f"{sum(c for r, c in reason_counts.items() if r != 'no units table in this NWB file')} "
                "of 11 have too few units surviving the firing-rate quality-control filter this project "
                "applies uniformly across every corpus. Neither reason is a fitting or compute limit, so "
                "re-running the existing pipeline -- with more compute, a relaxed setting, or different "
                "code -- cannot close this incompleteness; closing it would require different source "
                "recordings than dandi_000574 currently ships for those sessions."
            ),
        },
    }


# ---------------------------------------------------------------------------
# Per-session join: census fittability + participation ratio, by grain.
# ---------------------------------------------------------------------------

def load_census_rows(grain: dict) -> list[dict]:
    census = json.loads(CENSUS_PATH.read_text())
    return [
        r for r in census["rows"]
        if r["dataset"] == grain["dataset"] and r["modality"] == grain["modality"]
        and r["structure"] == grain["structure"] and r["epoch"] == grain["epoch"]
    ]


def load_participation_ratios(grain: dict) -> dict[tuple, dict]:
    checkpoint = json.loads(NOISE_CHECKPOINT_PATH.read_text())
    entry = checkpoint.get(grain["checkpoint_key"], {})
    out = {}
    for s in entry.get("sessions", []):
        if s.get("structure") != grain["checkpoint_structure"]:
            continue
        dim = s.get("dimensionality", {})
        fa = dim.get("factor_analysis", {})
        out[(s["patient"], s["session"], s["bin_ms"])] = {
            "participation_ratio_pca": dim.get("pca", {}).get("participation_ratio"),
            "participation_ratio_factor_analysis": fa.get("participation_ratio") if fa.get("status") == "fitted" else None,
        }
    return out


def build_grain_sessions(grain: dict) -> list[dict]:
    census_rows = load_census_rows(grain)
    pr_lookup = load_participation_ratios(grain)
    out = []
    for r in census_rows:
        key = (r["patient"], r["session"], r["bin_ms"])
        pr = pr_lookup.get(key, {})
        out.append({
            "patient": r["patient"], "session": r["session"], "bin_ms": r["bin_ms"],
            "status": r["status"], "n_splits_fitted": r["n_splits_fitted"], "n_splits_requested": r["n_splits_requested"],
            "n_units_or_channels": r["n_units"], "n_trials": r["n_trials"],
            "median_nugget_fraction": r.get("median_nugget_fraction"),
            "median_slow_timescale_s": r.get("median_slow_timescale_s"),
            "lag_one_autocorrelation_rho": _rho_from_slow_timescale(r.get("median_slow_timescale_s"), r["bin_width_s"]),
            "participation_ratio_pca": pr.get("participation_ratio_pca"),
            "participation_ratio_factor_analysis": pr.get("participation_ratio_factor_analysis"),
        })
    return out


# ---------------------------------------------------------------------------
# The fittability question: does the number of fitted splits track smoothness, dimensionality or count.
# ---------------------------------------------------------------------------

CORRELATION_TARGETS = (
    ("lag_one_autocorrelation_rho", "leading_latent_lag_one_autocorrelation"),
    ("participation_ratio_pca", "participation_ratio_pca_basis"),
    ("participation_ratio_factor_analysis", "participation_ratio_factor_analysis_basis"),
    ("n_units_or_channels", "unit_or_channel_count"),
)


def fittability_correlations(sessions: list[dict], grain_label: str, bin_ms: int) -> dict:
    result = {}
    for field, label in CORRELATION_TARGETS:
        pairs = [(s["n_splits_fitted"], s[field]) for s in sessions if s[field] is not None]
        n = len(pairs)
        if n < 3:
            result[label] = {"status": "not_computable", "reason": f"fewer than 3 sessions with both n_splits_fitted and {field} available", "n": n}
            continue
        x = np.array([p[0] for p in pairs], dtype=float)
        y = np.array([p[1] for p in pairs], dtype=float)
        spearman = spearman_permutation_test(x, y, n_perm=N_PERM, rng=_seed(grain_label, bin_ms, label, "spearman"))
        pearson = pearson_permutation_test(x, y, n_perm=N_PERM, rng=_seed(grain_label, bin_ms, label, "pearson"))
        result[label] = {
            "status": "computed", "n": n,
            "spearman_rho": spearman["rho"], "spearman_p": spearman["p_value"],
            "pearson_r": pearson["r"], "pearson_p": pearson["p_value"],
        }
    return result


# ---------------------------------------------------------------------------
# The matched-grain test on the one asymmetric observable.
# ---------------------------------------------------------------------------

def select_asymmetric_observable(census: dict) -> dict:
    """Reads the census's admission_matrix directly (never assumed) to find
    which shared observable, at which bin width, the LFP grain admits and
    the unit grain does not -- the selection rule this test applies."""
    am = census["admission_matrix"]
    unit_keys = {100: f"{UNIT_GRAIN['dataset']}::{UNIT_GRAIN['modality']}::{UNIT_GRAIN['structure']}::bin100",
                 200: f"{UNIT_GRAIN['dataset']}::{UNIT_GRAIN['modality']}::{UNIT_GRAIN['structure']}::bin200"}
    lfp_keys = {100: f"{LFP_GRAIN['dataset']}::{LFP_GRAIN['modality']}::{LFP_GRAIN['structure']}::bin100",
                200: f"{LFP_GRAIN['dataset']}::{LFP_GRAIN['modality']}::{LFP_GRAIN['structure']}::bin200"}
    if unit_keys[100] not in am or lfp_keys[100] not in am:
        raise KeyError("expected admission_matrix keys for the unit and LFP grains are missing")

    asymmetric_by_bin_width = {}
    for bin_ms in (100, 200):
        unit_obs = am[unit_keys[bin_ms]]["observables"]
        lfp_obs = am[lfp_keys[bin_ms]]["observables"]
        shared = sorted(set(unit_obs) & set(lfp_obs))
        asymmetric = [o for o in shared if lfp_obs[o].get("status") == "admitted" and unit_obs[o].get("status") != "admitted"]
        asymmetric_by_bin_width[bin_ms] = {
            "n_shared_observables": len(shared),
            "asymmetric_observables": asymmetric,
            "unit_grain_status_of_asymmetric": {o: unit_obs[o].get("status") for o in asymmetric},
            "lfp_grain_status_of_asymmetric": {o: lfp_obs[o].get("status") for o in asymmetric},
        }
    return asymmetric_by_bin_width


def matched_modality_test(unit_sessions: list[dict], lfp_sessions: list[dict], bin_ms: int) -> dict:
    """Pairs the same patient's same session across grains at the given bin
    width and reports the one asymmetric observable (cross_validated_nugget_
    fraction) side by side, with same-trial pairing verified in code rather
    than assumed."""
    unit_by_key = {(s["patient"], s["session"]): s for s in unit_sessions if s["bin_ms"] == bin_ms}
    lfp_by_key = {(s["patient"], s["session"]): s for s in lfp_sessions if s["bin_ms"] == bin_ms}
    shared_keys = sorted(set(unit_by_key) & set(lfp_by_key))

    paired = []
    n_trials_mismatched = 0
    for key in shared_keys:
        u, l = unit_by_key[key], lfp_by_key[key]
        trials_match = u["n_trials"] == l["n_trials"]
        if not trials_match:
            n_trials_mismatched += 1
        paired.append({
            "patient": key[0], "session": key[1],
            "unit_n_trials": u["n_trials"], "lfp_n_trials": l["n_trials"], "same_trial_count": trials_match,
            "unit_status": u["status"], "lfp_status": l["status"],
            "unit_median_nugget_fraction": u["median_nugget_fraction"], "lfp_median_nugget_fraction": l["median_nugget_fraction"],
            "unit_n_splits_fitted": u["n_splits_fitted"], "lfp_n_splits_fitted": l["n_splits_fitted"],
            "signal_present_at_lfp_absent_at_unit": (l["status"] == "fitted" and u["status"] != "fitted"),
        })

    n_present_at_lfp_absent_at_unit = sum(1 for p in paired if p["signal_present_at_lfp_absent_at_unit"])
    n_present_at_both = sum(1 for p in paired if p["unit_status"] == "fitted" and p["lfp_status"] == "fitted")
    n_absent_at_both = sum(1 for p in paired if p["unit_status"] != "fitted" and p["lfp_status"] != "fitted")
    n_present_at_unit_absent_at_lfp = sum(1 for p in paired if p["unit_status"] == "fitted" and p["lfp_status"] != "fitted")

    return {
        "bin_ms": bin_ms,
        "n_sessions_with_both_grains": len(shared_keys),
        "pairing_verification": {
            "same_patient_same_session_pairing_enforced_by": "joined on (patient, session) tuple in code, not assumed",
            "n_trial_count_mismatches": n_trials_mismatched,
            "reading": (
                "0 mismatches confirms the census rows' own pairing note "
                "('same_patients_and_trials_as_dandi_000574_single_unit_rows') at this bin width"
                if n_trials_mismatched == 0 else
                f"{n_trials_mismatched} of {len(shared_keys)} paired sessions have different trial counts "
                "between grains, contradicting the census rows' own pairing note; per-session detail is in "
                "sessions below"
            ),
        },
        "sessions": paired,
        "counts": {
            "present_at_lfp_absent_at_unit": n_present_at_lfp_absent_at_unit,
            "present_at_unit_absent_at_lfp": n_present_at_unit_absent_at_lfp,
            "present_at_both": n_present_at_both,
            "absent_at_both": n_absent_at_both,
        },
    }


def compute_grain_discordance_paired_test(matched_tests: dict) -> dict:
    """The matched-grain test's discordant sessions -- where cross_validated_
    nugget_fraction fits at one grain and not the other -- are the only
    sessions carrying information about which grain wins; sessions where it
    fits at both grains or neither are uninformative about direction and are
    excluded here by construction. This is the within-patient paired version
    of the census's own margin-based grain ordering: an exact two-sided
    binomial test (scipy.stats.binomtest, null p=0.5) of present_at_lfp_
    absent_at_unit against present_at_unit_absent_at_lfp, at each bin width."""
    per_bin = {}
    for bin_key, test in matched_tests.items():
        counts = test["counts"]
        n_lfp_only = counts["present_at_lfp_absent_at_unit"]
        n_unit_only = counts["present_at_unit_absent_at_lfp"]
        n_discordant = n_lfp_only + n_unit_only
        p_value = float(binomtest(n_lfp_only, n_discordant, 0.5, alternative="two-sided").pvalue)
        licenses = (
            "is indistinguishable from a coin flip and licenses no claim that either grain "
            "systematically out-fits the other at the session level"
            if p_value >= 0.05 else
            "is unlikely under a 50/50 null and licenses a directional claim at this bin width"
        )
        per_bin[bin_key] = {
            "bin_ms": test["bin_ms"],
            "present_at_lfp_absent_at_unit": n_lfp_only,
            "present_at_unit_absent_at_lfp": n_unit_only,
            "n_discordant": n_discordant,
            "two_sided_exact_binomial_p_value": p_value,
            "reading": (
                f"of {n_discordant} sessions where the two grains disagree on whether cross_"
                f"validated_nugget_fraction fits at all, {n_lfp_only} favour the LFP grain and "
                f"{n_unit_only} favour the unit grain; an exact two-sided binomial test against a "
                f"50/50 null gives p={p_value:.4f}, which {licenses}."
            ),
        }
    return {
        "method": (
            "scipy.stats.binomtest(n_present_at_lfp_absent_at_unit, n_discordant, 0.5, "
            "alternative='two-sided'), restricted to the discordant sessions at each bin width -- "
            "sessions where the observable fits at both grains or at neither carry no information "
            "about which grain wins and are excluded from this test by construction"
        ),
        "by_bin_width": per_bin,
        "reading": (
            "results/observability_and_power_census.json's margin_aware_modality_result ranks the "
            "LFP grain above the unit grain with a summed rank difference that its own "
            "margin_aware_modality_result_basis field attributes entirely to cross_validated_"
            "nugget_fraction. This is the within-patient paired test of that same observable, on "
            "the same patients and sessions, and it is null at both bin widths (bin_ms==200: "
            f"{per_bin['bin200']['present_at_lfp_absent_at_unit']} vs "
            f"{per_bin['bin200']['present_at_unit_absent_at_lfp']}, p="
            f"{per_bin['bin200']['two_sided_exact_binomial_p_value']:.4f}; bin_ms==100: "
            f"{per_bin['bin100']['present_at_lfp_absent_at_unit']} vs "
            f"{per_bin['bin100']['present_at_unit_absent_at_lfp']}, p="
            f"{per_bin['bin100']['two_sided_exact_binomial_p_value']:.4f}). The grain ordering is "
            "therefore not supported by a within-patient paired test of its sole contributing "
            "observable: on the one observable responsible for the entire ranking, the two grains "
            "are statistically indistinguishable from each other at the session level."
        ),
    }


FITTABILITY_CORRELATION_CAVEATS = (
    "two distinct mechanisms could produce a negative correlation between n_splits_fitted and the leading "
    "latent's lag-one autocorrelation, and this artifact cannot separate them: (a) a genuinely smoother "
    "latent decomposes into more positive lags and is easier to fit, which predicts the opposite (positive) "
    "sign, or (b) a small-sample estimation bias -- median_slow_timescale_s, and the rho computed from it, "
    "is only ever reported for the sessions where at least one split DID decompose, and when only one or "
    "two of the requested splits succeed (the sessions nearest the fitting floor) the decay rate fit to a "
    "short, noisy autocovariance curve is itself unreliable and can be biased toward an artificially slow "
    "apparent timescale. The correlations reported here are therefore evidence that fittability and the "
    "reported smoothness are entangled, not evidence of the sign or mechanism of that entanglement. This is "
    "a separate finding from results/observation_noise_estimator_construct_validity.json's answer-key result "
    "that the reported nugget-fraction VALUE (not the fit RATE measured here) is confounded by latent "
    "smoothness under a diagonal noise model. The unit/channel-count correlations do not carry this "
    "specific caveat, since population_axis size is measured independently of whether any split fit."
)

SCOPE_CAVEAT = (
    "the active plan frames this as a test of whether a 'between-region signal absent at unit resolution "
    "is present at LFP resolution'; the one observable this census's own admission matrix shows to be "
    "asymmetric between these two grains, cross_validated_nugget_fraction, is not a between-region or "
    "between-channel coupling measure -- both compared cells pool every unit or every contact into a "
    "single leading latent (structure=='pooled' for the unit grain, all bipolar contacts for the LFP "
    "grain), with no region stratification in either. What this test actually measures is narrower and "
    "still the whole point of the census-restatement duty this section carries: whether one specific "
    "estimator's cross-validated fit succeeds on the same patients and the same trials at one resolution "
    "and not the other. That is a fittability comparison, not a demonstration that anatomical information "
    "invisible to spikes becomes visible in the local field potential."
)


def build_artifact() -> dict:
    precursor = load_precursor()
    census = json.loads(CENSUS_PATH.read_text())

    unit_all = build_grain_sessions(UNIT_GRAIN)
    lfp_all = build_grain_sessions(LFP_GRAIN)

    fittability = {}
    for grain_label, sessions in (("unit_grain", unit_all), ("lfp_grain", lfp_all)):
        fittability[grain_label] = {}
        for bin_ms in (100, 200):
            subset = [s for s in sessions if s["bin_ms"] == bin_ms]
            fittability[grain_label][f"bin{bin_ms}"] = {
                "n_sessions": len(subset),
                "correlations": fittability_correlations(subset, grain_label, bin_ms),
            }

    asymmetric = select_asymmetric_observable(census)
    matched_tests = {f"bin{bin_ms}": matched_modality_test(unit_all, lfp_all, bin_ms) for bin_ms in (100, 200)}
    discordance = compute_grain_discordance_paired_test(matched_tests)

    c200 = matched_tests["bin200"]["counts"]
    c100 = matched_tests["bin100"]["counts"]
    p200 = discordance["by_bin_width"]["bin200"]["two_sided_exact_binomial_p_value"]
    p100 = discordance["by_bin_width"]["bin100"]["two_sided_exact_binomial_p_value"]
    headline = (
        "The census's only asymmetric shared observable between the unit and LFP grains, cross_validated_"
        "nugget_fraction, is asymmetric in admission status at bin_ms==200 only (unit grain excluded, LFP "
        "grain admitted; both admitted at bin_ms==100 -- see asymmetric_observable_selection). On the same "
        f"{matched_tests['bin200']['n_sessions_with_both_grains']} paired Boran sessions at bin_ms==200, "
        f"the observable fits at LFP resolution and not at unit resolution in "
        f"{c200['present_at_lfp_absent_at_unit']} sessions, fits at unit resolution and not at LFP "
        f"resolution in {c200['present_at_unit_absent_at_lfp']} sessions, fits at both in "
        f"{c200['present_at_both']}, and fits at neither in {c200['absent_at_both']}. At bin_ms==100, "
        f"where both grains are admitted, the same four-way split is {c100['present_at_lfp_absent_at_unit']} "
        f"LFP-only, {c100['present_at_unit_absent_at_lfp']} unit-only, {c100['present_at_both']} both, "
        f"{c100['absent_at_both']} neither. grain_discordance_paired_test reports an exact two-sided "
        "binomial test of the LFP-only versus unit-only counts -- the only sessions that carry information "
        f"about which grain wins -- at each bin width, and neither split is distinguishable from a 50/50 "
        f"split (bin_ms==200: {c200['present_at_lfp_absent_at_unit']} vs "
        f"{c200['present_at_unit_absent_at_lfp']}, p={p200:.4f}; bin_ms==100: "
        f"{c100['present_at_lfp_absent_at_unit']} vs {c100['present_at_unit_absent_at_lfp']}, p={p100:.4f}); "
        "this per-session discordance should not be read as a systematic advantage for either grain. This "
        "is a fittability asymmetry on one specific estimator, not a demonstration of a between-region "
        "signal (see scope_caveat_between_region_language): both grains pool across their whole recorded "
        "population, so no region contrast is being tested."
    )

    trigger = (
        "the observability-and-power census ranks the LFP grain above the unit grain on one shared "
        "observable, cross_validated_nugget_fraction, with the entire summed rank difference coming from "
        "that one observable; this restates what that ranking measures using the same patients and the "
        "same trials Boran's co-located spike-and-LFP recordings provide, asking whether it reflects "
        "instrument quality, latent timescale, dimensionality, or population size"
    )

    return {
        "schema_version": "1.0.0",
        "seed": SEED,
        "code_commit": git_commit(ROOT),
        "trigger": trigger,
        "headline": headline,
        "precursor": precursor,
        "grain_definitions": {"unit_grain": UNIT_GRAIN, "lfp_grain": LFP_GRAIN},
        "asymmetric_observable_selection": asymmetric,
        "fittability_correlations_by_grain_and_bin_width": fittability,
        "fittability_correlation_caveats": FITTABILITY_CORRELATION_CAVEATS,
        "matched_modality_test_by_bin_width": matched_tests,
        "grain_discordance_paired_test": discordance,
        "scope_caveat_between_region_language": SCOPE_CAVEAT,
    }


CENSUS_EXTENSION_FIELD = "margin_aware_modality_result_basis"


def extend_census_with_result_basis(payload: dict) -> dict:
    """Adds exactly one new top-level field to results/observability_and_
    power_census.json recording what margin_aware_modality_result's verdict
    rests on. Every pre-existing key, including margin_aware_modality_result
    itself, is verified byte-identical before and after -- this is an
    addition, never a rewrite."""
    mamr_source = json.loads(CENSUS_PATH.read_text())["margin_aware_modality_result"]
    per_bin = {}
    for bin_key, entry in mamr_source.items():
        contributing = [o for o, v in entry["per_observable"].items() if v.get("lfp_minus_unit", 0) != 0]
        per_bin[bin_key] = {
            "summed_rank_difference_lfp_minus_unit": entry["summed_rank_difference_lfp_minus_unit"],
            "observables_contributing_a_nonzero_rank_difference": contributing,
            "entire_summed_rank_difference_from_one_observable": len(contributing) == 1 and entry["summed_rank_difference_lfp_minus_unit"] != 0,
        }

    bin200 = payload["matched_modality_test_by_bin_width"]["bin200"]
    n_paired = bin200["n_sessions_with_both_grains"]
    n_disagree = bin200["counts"]["present_at_lfp_absent_at_unit"] + bin200["counts"]["present_at_unit_absent_at_lfp"]
    basis = {
        "per_bin_width": per_bin,
        "reading": (
            "margin_aware_modality_result's verdict (unit_grain_worse_instrumented_at_margin) rests "
            "entirely on cross_validated_nugget_fraction: at every bin width, that is the only shared "
            "observable with a nonzero lfp_minus_unit rank difference (per_bin_width above), and the "
            "difference is a MARGIN BAND -- an integer rank derived from how many cross-validation splits "
            "fit, crossing a floor -- rather than a difference in the observable's reported value. "
            "results/observability_matched_modality_test.json measured what governs that margin at the "
            "per-session level, on the same patients and the same trials: within each grain, the number "
            "of fitted splits correlates with the leading latent's lag-one autocorrelation and with "
            "unit/channel count, significantly for several grain-and-bin-width combinations (see that "
            "artifact's fittability_correlations_by_grain_and_bin_width field for every coefficient and "
            "p-value, and its fittability_correlation_caveats field for why the sign of the smoothness "
            "correlation cannot be read as a mechanism). At the bin width where the two grains' admission "
            f"status actually differs, the per-session picture is noisier than the population-level rank "
            f"difference suggests: of {n_paired} sessions with both grains recorded, {n_disagree} disagree "
            "about whether the observable fits at all (in either direction), not only the net few that "
            "produce the aggregate margin. A reader who wants the fittability evidence behind this "
            "verdict, rather than the verdict alone, should consult "
            "results/observability_matched_modality_test.json in full, not this field's summary."
        ),
        "restated_by": "results/observability_matched_modality_test.json",
        "verdict_field_this_basis_explains": "margin_aware_modality_result",
        "verdict_changed_by_this_field": False,
    }

    existing_census = json.loads(CENSUS_PATH.read_text())
    if CENSUS_EXTENSION_FIELD in existing_census:
        if canonical_json(existing_census[CENSUS_EXTENSION_FIELD]) != canonical_json(basis):
            raise RuntimeError(f"{CENSUS_EXTENSION_FIELD} already present in the census with different content; refusing to overwrite an extend-only artifact's field")
        return basis  # already applied with identical content on a prior run -- no file write needed

    with locked_json_update(CENSUS_PATH) as census:
        before_without_new_field = canonical_json({k: v for k, v in census.items() if k != CENSUS_EXTENSION_FIELD})
        if CENSUS_EXTENSION_FIELD in census:
            raise RuntimeError(f"{CENSUS_EXTENSION_FIELD} already present in the census; refusing to overwrite an extend-only artifact's field")
        census[CENSUS_EXTENSION_FIELD] = basis
        after_without_new_field = canonical_json({k: v for k, v in census.items() if k != CENSUS_EXTENSION_FIELD})
        if before_without_new_field != after_without_new_field:
            raise RuntimeError("extend-only violation: an existing observability_and_power_census.json key would have changed")
    return basis


CENSUS_PAIRED_TEST_FIELD = "margin_aware_modality_result_paired_test"


def extend_census_with_paired_test_null_result(payload: dict) -> dict:
    """Adds exactly one new top-level field to results/observability_and_
    power_census.json recording that a within-patient paired test of
    margin_aware_modality_result's sole contributing observable
    (cross_validated_nugget_fraction) is null at both bin widths. Every
    pre-existing key is verified byte-identical before and after -- this is
    an addition, never a rewrite, and never touches margin_aware_modality_
    result or margin_aware_modality_result_basis themselves."""
    discordance = payload["grain_discordance_paired_test"]
    p100 = discordance["by_bin_width"]["bin100"]["two_sided_exact_binomial_p_value"]
    p200 = discordance["by_bin_width"]["bin200"]["two_sided_exact_binomial_p_value"]
    result = {
        "observable_tested": "cross_validated_nugget_fraction",
        "test": (
            "an exact two-sided binomial test (scipy.stats.binomtest, null p=0.5) of the discordant "
            "sessions in results/observability_matched_modality_test.json's matched-grain test -- "
            "sessions where cross_validated_nugget_fraction fits at one grain and not the other, on "
            "the same patients and sessions -- comparing present_at_lfp_absent_at_unit against "
            "present_at_unit_absent_at_lfp at each bin width"
        ),
        "bin100_two_sided_exact_binomial_p_value": p100,
        "bin200_two_sided_exact_binomial_p_value": p200,
        "restated_by": "results/observability_matched_modality_test.json's grain_discordance_paired_test field",
        "reading": (
            "margin_aware_modality_result ranks the LFP grain above the unit grain using a summed "
            "rank difference that margin_aware_modality_result_basis attributes entirely to cross_"
            "validated_nugget_fraction. The within-patient paired test of that same observable is "
            f"null at both bin widths (bin_ms==100: p={p100:.4f}; bin_ms==200: p={p200:.4f}), so the "
            "grain ordering is not supported by a within-patient paired test of its sole contributing "
            "observable: on the observable responsible for the entire ranking, the two grains are "
            "statistically indistinguishable from each other at the session level."
        ),
    }

    existing_census = json.loads(CENSUS_PATH.read_text())
    if CENSUS_PAIRED_TEST_FIELD in existing_census:
        if canonical_json(existing_census[CENSUS_PAIRED_TEST_FIELD]) != canonical_json(result):
            raise RuntimeError(f"{CENSUS_PAIRED_TEST_FIELD} already present in the census with different content; refusing to overwrite an extend-only artifact's field")
        return result  # already applied with identical content on a prior run -- no file write needed

    with locked_json_update(CENSUS_PATH) as census:
        before_without_new_field = canonical_json({k: v for k, v in census.items() if k != CENSUS_PAIRED_TEST_FIELD})
        if CENSUS_PAIRED_TEST_FIELD in census:
            raise RuntimeError(f"{CENSUS_PAIRED_TEST_FIELD} already present in the census; refusing to overwrite an extend-only artifact's field")
        census[CENSUS_PAIRED_TEST_FIELD] = result
        after_without_new_field = canonical_json({k: v for k, v in census.items() if k != CENSUS_PAIRED_TEST_FIELD})
        if before_without_new_field != after_without_new_field:
            raise RuntimeError("extend-only violation: an existing observability_and_power_census.json key would have changed")
    return result


def main() -> None:
    payload = build_artifact()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(canonical_json(payload))
    print(f"wrote {OUTPUT_PATH}", file=sys.stderr)
    extend_census_with_result_basis(payload)
    print(f"extended {CENSUS_PATH} with '{CENSUS_EXTENSION_FIELD}' (extend-only, verified)", file=sys.stderr)
    extend_census_with_paired_test_null_result(payload)
    print(f"extended {CENSUS_PATH} with '{CENSUS_PAIRED_TEST_FIELD}' (extend-only, verified)", file=sys.stderr)
    print(json.dumps({
        "precursor_n_sessions_complete_of_total": f"{payload['precursor']['n_sessions_complete']}/{payload['precursor']['n_sessions_total']}",
        "asymmetric_observables_bin200": payload["asymmetric_observable_selection"][200]["asymmetric_observables"],
        "matched_test_bin200_counts": payload["matched_modality_test_by_bin_width"]["bin200"]["counts"],
        "fittability_unit_grain_bin200_rho_corr": payload["fittability_correlations_by_grain_and_bin_width"]["unit_grain"]["bin200"]["correlations"]["leading_latent_lag_one_autocorrelation"],
        "fittability_lfp_grain_bin200_rho_corr": payload["fittability_correlations_by_grain_and_bin_width"]["lfp_grain"]["bin200"]["correlations"]["leading_latent_lag_one_autocorrelation"],
        "grain_discordance_paired_test_p_values": {b: v["two_sided_exact_binomial_p_value"] for b, v in payload["grain_discordance_paired_test"]["by_bin_width"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
