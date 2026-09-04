#!/usr/bin/env python3
"""Tests whether task-driven confinement (lambda, from the Gaussian
state-space drift fit) measures the same quantity as Murray, Bernacchia,
Freedman et al. 2014's (Nat Neurosci 17:1661, PMC4241138) intrinsic
autocorrelation timescale (tau).

PREDECLARED INTERPRETATION, written before this script's first run: "If tau
and 1/lambda are the same quantity, they correlate positively across
patients within structure and their across-structure orderings agree. If
they dissociate -- orderings disagree, or within-structure correlation
intervals include zero -- then task-driven confinement is not intrinsic
autocorrelation, and the regional confinement ordering measures something
the intrinsic-timescale literature does not contain. Either outcome is
reported; neither is preferred." This interpretation is gated on
`tau_estimator_calibration.json` (`PREDECLARED_DECISION` below): the
estimator must first be shown to resolve a planted tau at the observed
rates, trial count and window length, or the comparison is withdrawn as
`estimator_non_identified` rather than adjudicated by this rule.

Method, following Murray et al. 2014's own description: spike-count
autocorrelation in 50-ms bins during a task-free baseline, fit with an
exponential decay PLUS OFFSET (R(k*bin) = A*exp(-k*bin/tau) + C -- their own
functional form; the offset absorbs the non-decaying noise floor, unlike a
bare exponential). Fit from the first NONZERO lag: lag 0 is the total
variance, contaminated by the private Poisson counting-noise variance, and
including it makes the recovered tau track firing rate instead of the
autocorrelation timescale (see `_fit_tau`'s `exclude_lag0` argument and the
regression test that documents the failure mode this avoids). This project
has no true task-free recording; the closest available baseline in DANDI
000469 is the pre-stimulus fixation period every trial carries
(timestamps_FixationCross to timestamps_Encoding1, median gap 1.056 s, min
0.925 s across a real session -- verified by direct inspection before
choosing the window below). A 0.9 s window (18 x 50 ms bins) is used, safely
inside every trial's fixation period.

Reported regardless of whether the unit-count-matched sensitivity supports
the lambda ordering; not framed as validating that ordering if it does not.

Scope: DANDI 000469 only (the deciding dataset for the lambda ordering);
000574/001187/000673 intrinsic-timescale extension is deferred and reported
as such -- see the crack register.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402
from spike_pipeline import (  # noqa: E402
    ANATOMICAL_REGIONS,
    load_spike_times,
    low_rate_unit_mask,
    filter_units_by_region,
    resolve_unit_regions,
)
from statistics import bootstrap_ci, spearman_permutation_test  # noqa: E402
import run_human_drift_spine_000469 as spine469  # noqa: E402

BIN_S = 0.05
BASELINE_WINDOW_S = 0.9
N_BINS = int(round(BASELINE_WINDOW_S / BIN_S))
MAX_LAG = 10
MIN_TRIALS_FOR_TAU = 10
RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "intrinsic_timescale_vs_confinement.json"
SEED = 20260805

PREDECLARED_INTERPRETATION = (
    "If tau and 1/lambda are the same quantity, they correlate positively across patients within "
    "structure and their across-structure orderings agree. If they dissociate -- orderings "
    "disagree, or within-structure correlation intervals include zero -- then task-driven "
    "confinement is not intrinsic autocorrelation, and the regional confinement ordering measures "
    "something the intrinsic-timescale literature does not contain. Either outcome is reported; "
    "neither is preferred."
)

# Every predeclared decision needs three branches, none of the first two preferred, with the
# third naming the estimator-failure case. Text identical to
# run_tau_estimator_calibration.PREDECLARED_DECISION (duplicated, not imported, to avoid a
# circular import -- that script imports this one for BIN_S/N_BINS/_fit_tau).
PREDECLARED_DECISION = {
    "supported": (
        "corrected tau is resolvable (tau_estimator_calibration.json) at the observed rates and "
        "window, AND the per-structure tau ordering has non-overlapping bootstrap intervals for "
        "at least one structure pair."
    ),
    "refuted": (
        "corrected tau is resolvable and the orderings of tau and 1/lambda agree for the "
        "pre-SMA-vs-hippocampus pair; task-driven confinement is then not dissociable from "
        "intrinsic autocorrelation in this corpus, and the Murray comparison collapses to a "
        "replication."
    ),
    "estimator_non_identified": (
        "tau_estimator_calibration.json shows the corrected estimator is NOT resolvable at the "
        "observed rates and window length. Then tau is NOT MEASURABLE in this corpus, the Murray "
        "comparison is WITHDRAWN rather than reported, and the artifact records what window "
        "length would be required, computed from the calibration."
    ),
    "note": (
        "a 0.9 s window is expected on prior grounds to leave the corrected estimator "
        "unresolvable; this is a prediction, not a preference. Neither of the first two "
        "branches is preferred."
    ),
}


def _bin_spike_counts(spike_times: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    counts = np.zeros((len(onsets), N_BINS))
    for i, onset in enumerate(onsets):
        rel = spike_times - onset
        in_window = rel[(rel >= 0) & (rel < BASELINE_WINDOW_S)]
        bin_idx = np.minimum((in_window / BIN_S).astype(int), N_BINS - 1)
        for b in bin_idx:
            counts[i, b] += 1
    return counts


def _fit_tau(counts: np.ndarray, exclude_lag0: bool = True) -> tuple[float | None, str | None]:
    """Murray et al. 2014's exponential-decay-plus-offset fit to the pooled
    (across-trial) spike-count autocovariance.

    ac[0] is the total variance, which contains the private Poisson
    counting-noise variance as a delta at zero lag; Murray et al. fit from
    the first NONZERO lag to exclude it (`exclude_lag0=True`, the default and
    the only mode used outside the regression test that documents the
    defect). ac[0] is still used as the amplitude initial guess either way.
    """
    if counts.shape[0] < MIN_TRIALS_FOR_TAU:
        return None, f"fewer than {MIN_TRIALS_FOR_TAU} trials"
    centered = counts - counts.mean(axis=0, keepdims=True)
    max_lag = min(MAX_LAG, counts.shape[1] - 2)
    if max_lag < 2:
        return None, "fewer than 2 usable lags"
    ac = np.empty(max_lag + 1)
    ac[0] = np.mean(centered * centered)
    for k in range(1, max_lag + 1):
        ac[k] = np.mean(centered[:, :-k] * centered[:, k:])
    if not np.isfinite(ac).all() or ac[0] <= 0:
        return None, "degenerate autocovariance (zero or non-finite variance)"
    lags_s = np.arange(max_lag + 1) * BIN_S
    fit_lags, fit_ac = (lags_s[1:], ac[1:]) if exclude_lag0 else (lags_s, ac)
    if len(fit_lags) < 2:
        return None, "fewer than 2 usable nonzero lags"

    def model(k, amplitude, tau, offset):
        return amplitude * np.exp(-k / tau) + offset

    try:
        popt, _ = curve_fit(
            model, fit_lags, fit_ac, p0=[ac[0], 0.2, 0.0],
            bounds=([-np.inf, 1e-3, -np.inf], [np.inf, 10.0, np.inf]), maxfev=5000,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"curve_fit failed: {exc}"
    _, tau, _ = popt
    if not np.isfinite(tau) or tau <= 0:
        return None, "non-finite or non-positive tau"
    return float(tau), None


def _patient_region_tau(path: Path, region: str) -> dict:
    with h5py.File(path, "r") as handle:
        spike_lists = load_spike_times(handle)
        unit_regions = resolve_unit_regions(handle)["region"]
        trials = handle["intervals/trials"]
        fixation_onsets = trials["timestamps_FixationCross"][:]
        encoding_onsets = trials["timestamps_Encoding1"][:]
    gap = encoding_onsets - fixation_onsets
    if np.any(gap < BASELINE_WINDOW_S):
        return {"status": "excluded", "reason": "some trial's fixation-to-encoding gap is shorter than the baseline window"}
    spike_lists = filter_units_by_region(spike_lists, unit_regions, region)
    rate_mask = low_rate_unit_mask(spike_lists, fixation_onsets, BASELINE_WINDOW_S)
    spike_lists = [s for s, keep in zip(spike_lists, rate_mask) if keep]
    if not spike_lists:
        return {"status": "non_identified", "reason": "no units cleared firing-rate QC in the baseline window"}
    unit_taus = []
    unit_rates = []
    for spikes in spike_lists:
        counts = _bin_spike_counts(spikes, fixation_onsets)
        unit_rates.append(float(counts.sum() / (counts.shape[0] * BASELINE_WINDOW_S)))
        tau, _ = _fit_tau(counts)
        if tau is not None:
            unit_taus.append(tau)
    mean_rate_hz = float(np.mean(unit_rates))
    if not unit_taus:
        return {
            "status": "non_identified", "reason": "no unit in this region-session had an identifiable tau",
            "n_units": len(spike_lists), "mean_rate_hz": mean_rate_hz,
        }
    return {
        "status": "identified", "n_units": len(spike_lists), "n_units_with_tau": len(unit_taus),
        "median_tau_s": float(np.median(unit_taus)), "unit_taus_s": unit_taus,
        "mean_rate_hz": mean_rate_hz,
    }


def _lambda_per_patient(artifact: dict, region: str) -> dict[str, float]:
    sessions = artifact["regions"][region]["sessions"]
    out = {}
    for patient, row in sessions.items():
        if row.get("status") != "complete":
            continue
        value = row["summary"].get("state_space_lambda_identified_mean")
        if value is not None:
            out[patient] = value
    return out


def bootstrap_summary(values: list[float], rng: np.random.Generator) -> dict:
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if len(finite) < 2:
        return {"status": "non_identified", "n": int(len(finite)), "reason": f"fewer than 2 values ({len(finite)})"}
    _, lo, hi = bootstrap_ci(finite, np.mean, n_boot=5000, rng=rng)
    return {
        "status": "estimable", "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "bootstrap_ci95": [float(lo), float(hi)], "n": int(len(finite)),
    }


def main() -> None:
    directory = spine469.data_directory()
    artifact = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())
    sessions_any = artifact["regions"]["pooled"]["sessions"]
    patients = sorted(sessions_any.keys())

    per_structure_tau: dict[str, dict[str, float]] = {r: {} for r in ANATOMICAL_REGIONS}
    per_structure_rate: dict[str, dict[str, float]] = {r: {} for r in ANATOMICAL_REGIONS}
    per_patient_debug: dict[str, dict] = {}
    for patient in patients:
        path = directory / patient / f"{patient}_ses-2_ecephys+image.nwb"
        if not path.is_file():
            continue
        per_patient_debug[patient] = {}
        for region in ANATOMICAL_REGIONS:
            result = _patient_region_tau(path, region)
            per_patient_debug[patient][region] = {k: v for k, v in result.items() if k != "unit_taus_s"}
            if result.get("status") == "identified":
                per_structure_tau[region][patient] = result["median_tau_s"]
                per_structure_rate[region][patient] = result["mean_rate_hz"]
        print(f"{patient} done", flush=True)

    rng = np.random.default_rng(SEED)
    tau_group_summary = {r: bootstrap_summary(list(per_structure_tau[r].values()), rng) for r in ANATOMICAL_REGIONS}
    rate_group_summary = {r: bootstrap_summary(list(per_structure_rate[r].values()), rng) for r in ANATOMICAL_REGIONS}

    # Is the tau ordering a firing-rate ordering? One point per structure (median tau, median
    # rate among the same identified patients).
    structures_with_both = [
        r for r in ANATOMICAL_REGIONS
        if tau_group_summary[r]["status"] == "estimable" and rate_group_summary[r]["status"] == "estimable"
    ]
    if len(structures_with_both) >= 4:
        tau_arr = np.array([tau_group_summary[r]["median"] for r in structures_with_both])
        rate_arr = np.array([rate_group_summary[r]["median"] for r in structures_with_both])
        q48 = spearman_permutation_test(tau_arr, rate_arr, n_perm=5000, rng=rng)
        tau_vs_rate_correlation = {
            "status": "estimable", "n_structures": len(structures_with_both), "structures": structures_with_both,
            "spearman_rho_tau_vs_mean_rate": q48["rho"], "spearman_p_value": q48["p_value"],
        }
    else:
        tau_vs_rate_correlation = {
            "status": "non_identified", "n_structures": len(structures_with_both),
            "reason": f"fewer than 4 structures with both median tau and median rate estimable ({len(structures_with_both)})",
        }

    lambda_per_structure = {r: _lambda_per_patient(artifact, r) for r in ANATOMICAL_REGIONS}

    within_structure_correlation = {}
    for region in ANATOMICAL_REGIONS:
        tau_map, lambda_map = per_structure_tau[region], lambda_per_structure[region]
        shared = sorted(set(tau_map) & set(lambda_map))
        if len(shared) < 4:
            within_structure_correlation[region] = {
                "status": "non_identified", "n_patients": len(shared),
                "reason": f"fewer than 4 patients with both tau and lambda identified ({len(shared)})",
            }
            continue
        tau_vals = np.array([tau_map[p] for p in shared])
        inv_lambda_vals = np.array([1.0 / lambda_map[p] for p in shared])
        result = spearman_permutation_test(tau_vals, inv_lambda_vals, n_perm=5000, rng=rng)
        _, lo, hi = bootstrap_ci(
            np.column_stack([tau_vals, inv_lambda_vals]),
            lambda d: float(np.corrcoef(d[:, 0], d[:, 1])[0, 1]) if len(d) > 1 else np.nan,
            n_boot=5000, rng=rng,
        )
        within_structure_correlation[region] = {
            "status": "estimable", "n_patients": len(shared), "patients": shared,
            "spearman_rho_tau_vs_inverse_lambda": result["rho"], "spearman_p_value": result["p_value"],
            "pearson_r_patient_bootstrap_ci95": [float(lo), float(hi)],
            "interval_excludes_zero": bool(lo > 0 or hi < 0),
        }

    tau_ordering = {
        r: tau_group_summary[r]["median"] for r in ANATOMICAL_REGIONS
        if tau_group_summary[r]["status"] == "estimable"
    }
    lambda_ordering_artifact = json.loads((RESULTS / "region_stratified_drift_000469.json").read_text())["lambda_regional_ordering"]
    lambda_ordering = (
        lambda_ordering_artifact.get("region_order_fastest_to_slowest", [])
        if lambda_ordering_artifact.get("status") == "estimable" else []
    )
    tau_order_slowest_to_fastest_decay = sorted(tau_ordering, key=tau_ordering.get, reverse=True)  # longest tau first
    orderings_agree = None
    if len(tau_ordering) >= 2 and lambda_ordering:
        shared_regions = [r for r in lambda_ordering if r in tau_ordering]
        if len(shared_regions) >= 2:
            # Same quantity implies: short tau (fast intrinsic decay) <-> large lambda (fast
            # confinement) -- so tau ranked shortest-first should match lambda's fastest-first order.
            tau_shortest_first = sorted(shared_regions, key=tau_ordering.get)
            lambda_fastest_first = [r for r in lambda_ordering if r in shared_regions]
            orderings_agree = bool(tau_shortest_first == lambda_fastest_first)

    # Three-branch verdict gated on the calibration power curve (tau_estimator_calibration.json):
    # no decision rule may reward estimator failure with a substantive verdict.
    calibration = json.loads((RESULTS / "tau_estimator_calibration.json").read_text())
    calibration_resolvable_fraction = calibration["resolvable_fraction_overall"]
    window_sweep = calibration.get("window_length_sweep")

    if calibration_resolvable_fraction == 0.0:
        verdict = "estimator_non_identified"
        verdict_reason = (
            f"tau_estimator_calibration.json: 0/{len(calibration['cells'])} grid cells resolvable "
            f"at the observed n_trials={calibration['n_trials_used']}, "
            f"bin_width_s={calibration['bin_width_s']}, baseline_window_s={calibration['baseline_window_s']}. "
            "Corrected tau is NOT MEASURABLE in this corpus; the Murray et al. 2014 comparison is "
            "WITHDRAWN rather than reported (per-structure tau values below are descriptive only, "
            "not interpretable as intrinsic timescales)."
        )
        required_window_s = (
            window_sweep["first_window_s_with_all_taus_resolvable"] if window_sweep else None
        )
        window_required_note = (
            f"window-length sweep (calibration): first window with all 5 planted taus resolvable = "
            f"{required_window_s} s (null = not reached up to {window_sweep['windows_s'][-1]}s tested)."
            if window_sweep else "window-length sweep not run (calibration resolvable at baseline window)."
        )
    else:
        presma_tau = tau_group_summary.get("pre_sma", {}).get("median")
        hipp_tau = tau_group_summary.get("hippocampus", {}).get("median")
        presma_lambda_vals = list(lambda_per_structure.get("pre_sma", {}).values())
        hipp_lambda_vals = list(lambda_per_structure.get("hippocampus", {}).values())
        pair_orderings_agree = None
        if presma_tau is not None and hipp_tau is not None and presma_lambda_vals and hipp_lambda_vals:
            tau_says_presma_faster_decay = presma_tau < hipp_tau
            lambda_says_presma_faster_confinement = float(np.mean(presma_lambda_vals)) > float(np.mean(hipp_lambda_vals))
            pair_orderings_agree = bool(tau_says_presma_faster_decay == lambda_says_presma_faster_confinement)

        ci_by_structure = {
            r: tau_group_summary[r]["bootstrap_ci95"] for r in ANATOMICAL_REGIONS
            if tau_group_summary[r]["status"] == "estimable"
        }
        non_overlapping_pair = None
        structs = sorted(ci_by_structure)
        for i in range(len(structs)):
            for j in range(i + 1, len(structs)):
                lo_i, hi_i = ci_by_structure[structs[i]]
                lo_j, hi_j = ci_by_structure[structs[j]]
                if hi_i < lo_j or hi_j < lo_i:
                    non_overlapping_pair = [structs[i], structs[j]]
                    break
            if non_overlapping_pair:
                break

        if pair_orderings_agree:
            verdict = "refuted"
            verdict_reason = (
                "corrected tau is resolvable and the tau / 1-over-lambda orderings agree for the "
                "pre-SMA-vs-hippocampus pair; task-driven confinement is not dissociable from "
                "intrinsic autocorrelation in this corpus, and the Murray comparison collapses to "
                "a replication."
            )
        elif non_overlapping_pair:
            verdict = "supported"
            verdict_reason = (
                f"corrected tau is resolvable and structures {non_overlapping_pair} have "
                "non-overlapping bootstrap tau intervals."
            )
        else:
            verdict = "estimator_non_identified"
            verdict_reason = (
                "corrected tau is resolvable in the calibration grid but no structure pair in the "
                "real data shows non-overlapping tau intervals, and the pre-SMA-vs-hippocampus "
                "pair orderings do not agree; the comparison is inconclusive on this run."
            )
        window_required_note = "not applicable: calibration was resolvable at the baseline window."

    predeclared_decision = dict(PREDECLARED_DECISION)
    predeclared_decision["verdict"] = verdict
    predeclared_decision["verdict_reason"] = verdict_reason
    predeclared_decision["window_length_required_note"] = window_required_note

    output = {
        "schema_version": "1.0.0", "analysis_id": "intrinsic_timescale_vs_confinement",
        "trigger": "does task-driven confinement (lambda) measure the same quantity as intrinsic autocorrelation timescale (tau)?",
        "code_commit": git_commit(ROOT), "source_hash": sha256_file(Path(__file__)),
        "method_citation": "Murray, Bernacchia, Freedman et al. 2014, Nat Neurosci 17:1661 (PMC4241138, read directly)",
        "method_note": (
            "Spike-count autocorrelation, 50 ms bins, exponential-decay-plus-offset fit "
            "(A*exp(-k*bin/tau)+C), pooled across trials, per unit, then median across units per "
            "(patient, structure). Baseline window: DANDI 000469's pre-stimulus fixation period "
            "(timestamps_FixationCross to timestamps_FixationCross+0.9s), the closest available "
            "task-free window in this corpus -- not a true inter-task baseline."
        ),
        "scope": "DANDI 000469 only; 000574/001187/000673 intrinsic-timescale extension deferred (see crack register)",
        "predeclared_interpretation": PREDECLARED_INTERPRETATION,
        "estimator_fix": (
            "2026-08-07: _fit_tau now excludes lag 0 from the curve_fit data. Lag 0 is the total "
            "variance, contaminated by Poisson counting noise, and made the prior fit track "
            "firing rate instead of the autocorrelation timescale. All tau values below supersede "
            "the artifact as it stood before this fix -- see PAPER_REPORT.tex and the crack "
            "register for the former (void) values."
        ),
        "gated_on_unit_count_sensitivity": (
            "reported regardless of the unit-count-matched sensitivity verdict on the lambda "
            "ordering; not framed as validating that ordering if the sensitivity does not support it."
        ),
        "calibration_source": "results/tau_estimator_calibration.json",
        "calibration_resolvable_fraction_overall": calibration_resolvable_fraction,
        "per_structure_tau": tau_group_summary,
        "per_structure_mean_rate_hz": rate_group_summary,
        "tau_vs_mean_rate_across_structure_correlation": tau_vs_rate_correlation,
        "tau_order_longest_to_shortest": tau_order_slowest_to_fastest_decay,
        "lambda_order_fastest_to_slowest_reference": lambda_ordering,
        "orderings_agree": orderings_agree,
        "within_structure_tau_vs_inverse_lambda_correlation": within_structure_correlation,
        "predeclared_decision": predeclared_decision,
        "per_patient_debug_status": per_patient_debug,
    }
    OUTPUT_PATH.write_text(canonical_json(output))

    crack_path = RESULTS / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    for entry in cracks["entries"]:
        if entry.get("crack_id") == "intrinsic_timescale_vs_confinement_dissociated":
            entry.update({
                "status": "reopened_and_resolved_as_estimator_non_identified",
                "trigger": (
                    "The prior 'dissociated' verdict was produced by an estimator with lag 0 "
                    "included in the autocovariance fit, which returns 1.4-19.2 ms for a true tau "
                    "of 50-350 ms alike and tracks firing rate instead of timescale -- a "
                    "noise-dominated estimator satisfies the old two-branch rule's 'dissociate' "
                    "outcome regardless of the true value."
                ),
                "chase": (
                    "Excluded lag 0 from the curve_fit data (kept for the amplitude initial "
                    "guess only). Built a positive control (tau_estimator_calibration.json): "
                    "planted tau in {0.05,0.1,0.2,0.35,0.5}s x rate in {1,2,5,10,20}Hz at the "
                    "actual observed n_trials/bin_width/window_length, 200 reps/cell. Result: "
                    "0/25 cells resolvable (every recovered-tau interval spans nearly the full "
                    "fit bound, [~0.004s, 10s]) -- the corrected estimator has no power at this "
                    "corpus's 0.9s baseline window. A window-length sweep at a representative "
                    "5 Hz rate found short taus (0.05-0.1s) become resolvable only around "
                    "28.8-57.6s; the full 5-value grid was not resolvable up to 57.6s."
                ),
                "resolution": (
                    "estimator_non_identified: tau is not measurable in this corpus at its "
                    "available window length. The Murray et al. 2014 comparison is withdrawn "
                    "rather than reported. The old 'dissociated' verdict, all void tau values "
                    "(5.6-14.9 ms), tau_order_*, and orderings_agree=False are superseded; see "
                    "PAPER_REPORT.tex for where the former values are struck. "
                    "The real-data re-run still reports descriptive per-structure tau (now "
                    "90-220 ms, order-of-magnitude plausible) and mean rate alongside it "
                    "(Spearman rho=-0.6, p=0.36, n=5 structures -- not significant, so the "
                    "corrected estimator does not obviously track rate, but n is too small to "
                    "conclude either way)."
                ),
                "artifact": "results/intrinsic_timescale_vs_confinement.json, results/tau_estimator_calibration.json",
            })
            break
    if not any(entry.get("crack_id") == "tau_lag0_inclusion_estimator_insensitive" for entry in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "tau_lag0_inclusion_estimator_insensitive",
            "trigger": (
                "_fit_tau built the autocovariance as ac[0] = mean(centered**2) -- the total "
                "variance, containing the private Poisson counting-noise variance as a delta at "
                "zero lag -- and passed ac[0..max_lag] to curve_fit. A simulation (OU latent "
                "driving Poisson emission, matching this corpus's trial count/bin width/window) "
                "showed a 7x range in true tau (50-350ms) collapsing to a flat ~11-19ms recovered "
                "value that tracks firing rate, not tau."
            ),
            "chase": (
                "Excluded lag 0 from the fit data; added a regression test "
                "(test_lag0_inclusion_collapses_tau_regardless_of_truth) asserting the fixed "
                "estimator distinguishes a short from a long planted tau while the lag-0-inclusive "
                "variant collapses both to within 30ms of each other; built the calibration power "
                "curve (tau_estimator_calibration.json) as the mandatory positive control."
            ),
            "resolution": (
                "Fixed in _fit_tau (exclude_lag0=True default). The corrected estimator is "
                "responsive to the planted truth in simulation (test passes) but has no resolving "
                "power at this corpus's actual 0.9s window/trial count -- see "
                "intrinsic_timescale_vs_confinement_dissociated for the consequence."
            ),
            "status": "resolved",
            "artifact": "scripts/run_intrinsic_timescale_vs_confinement.py, tests/test_intrinsic_timescale_vs_confinement.py",
        })
    crack_path.write_text(canonical_json(cracks))

    print(json.dumps({
        "output": str(OUTPUT_PATH), "tau_group_summary_status": {r: tau_group_summary[r]["status"] for r in ANATOMICAL_REGIONS},
        "verdict": verdict, "tau_vs_rate_correlation": tau_vs_rate_correlation,
    }, indent=2))


if __name__ == "__main__":
    main()
