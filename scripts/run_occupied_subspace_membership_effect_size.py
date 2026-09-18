"""run_occupied_subspace_membership_effect_size.py -- re-answers, as an ESTIMATION question rather than a
null-hypothesis test, whether the residual-direction axis lies inside the occupied state space, for the
two macaque corpora the delivered occupied-state-space decomposition covers.

THE DEFECT THIS REPLACES, verified numerically before anything below is trusted. The delivered rank
re-estimation validated two rank selectors on synthetic data and applied them to real cells, then asked
whether the axis's off-fraction was "significantly above" either of two nulls. One of those nulls --
`_within_subspace_null_decomposition` in the module this one reuses -- builds its null directions as
`coeffs @ basis.T`, a linear combination of the SAME basis vectors that span the tested subspace. Those
directions lie exactly inside that subspace by construction, so their off-fraction is zero up to
floating-point round-off (the delivered artifact records `null_off_fraction_mean` around 1e-17 and flags
`null_is_degenerate_by_construction: true` on every cell). Any strictly positive observed off-fraction --
which is every real cell, since no empirical direction is EXACTLY inside a data-driven subspace -- therefore
tests as "significantly above" this null with p-values near the floor of what a few hundred draws can
resolve. This one null being unable to fail was enough, under the delivered branch rule, to call the axis
"outside the occupied space" for both corpora, even though the SAME cells separately recorded
`within_fraction` between 0.9976 and 0.9994 against the harder, more informative comparison -- the ambient
null -- which those same cells passed (`off_fraction_above_null: false`). The two facts contradict a
hypothesis-testing framing but are perfectly consistent with an estimation framing: the axis sits almost
entirely inside a fairly small selected subspace, and "almost entirely" cannot be improved on by any
direction that is allowed to live only inside that same subspace.

THE ANALYTIC CHANCE REFERENCE THIS MODULE ADDS. For a FIXED k-dimensional subspace of R^p (any orthonormal
basis b_1..b_k, fixed independently of the direction being tested) and a direction v drawn UNIFORMLY at
random on the unit sphere in the full p-dimensional ambient space, extend {b_1..b_k} to a full orthonormal
basis {b_1..b_p} of R^p. Because ||v||^2 = sum_{j=1}^{p} (b_j . v)^2 = 1 for every unit vector regardless of
which orthonormal basis is used, and the uniform distribution on the sphere is invariant under orthogonal
transformations (rotating which basis vector is "first" does not change v's law), every term (b_j . v)^2
has the same expectation, 1/p. Summing the first k of them gives the expected within-subspace fraction:
    E[ sum_{j=1}^{k} (b_j . v)^2 ] = k / p.
This is the closed form of exactly what the delivered pipeline's own "ambient null" already estimates by
Monte Carlo (>=200 random ambient unit vectors, same fixed subspace) -- the two are cross-checked against
each other below as a sanity gate on this derivation, not asserted on faith.

THE EFFECT SIZE. Per corpus, per session, per item-count level, per estimator that PASSED the synthetic
recovery gate at that corpus's ambient scale (both estimators from `run_occupied_subspace_rank_estimation`,
reused unchanged, together with the identical gate that validated them there): `within_fraction - k/p`,
where `within_fraction` is the same quantity the delivered artifact already computed (the squared-norm
fraction of the residual-direction axis captured by the estimator-selected rank-k basis of the unit-
direction matrix) and `k/p` is the analytic chance reference above, using that cell's own selected rank k
and ambient unit count p. This contrast is the number this module treats as evidence; the two null
comparisons the delivered pipeline computed are carried forward and reported in full, but neither is used
to fire a branch -- see the per-null disclosures in `DECISION_RULE_DECLARED_BEFORE_FITTING` below.

WHAT IS REUSED, UNCHANGED. `entry_holdout_rank`, `permutation_eigenvalue_rank`, `_cell_under_estimator`,
`_run_corpus`, `run_synthetic_recovery_gate`, `_gate_status_for_corpus`, `_load_checkpoint`, `ESTIMATOR_NAMES`
(the two synthetically-gated rank estimators and their real-data application, from
`run_occupied_subspace_rank_estimation`, whose own on-disk checkpoints are reused directly since neither
estimator nor the data it runs on has changed); `_collect_axis_entries`, `leading_eigenvector`, `CORPORA`,
`N_RANDOM_AXIS_DRAWS`, `_trial_count_weighted` (from `run_deviation_axis_structure`); `_load_corpora_and_
accounting` (from `run_occupied_subspace_rank_selection_repair`); `N_BOOT_SESSION_CLUSTER`, `Z_80_POWER`
(from `run_component_identity_subspace_atlas`). The only new code here is the analytic chance reference,
the contrast and its whole-session cluster bootstrap, the reproduction-gate recomputation, and the
estimation-framed branch rule.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from run_component_identity_subspace_atlas import N_BOOT_SESSION_CLUSTER, Z_80_POWER  # noqa: E402
from run_deviation_axis_structure import (  # noqa: E402
    CORPORA, N_RANDOM_AXIS_DRAWS, _collect_axis_entries, _trial_count_weighted, leading_eigenvector,
)
from run_occupied_subspace_rank_estimation import (  # noqa: E402
    ESTIMATOR_NAMES, _cell_under_estimator, _gate_status_for_corpus, _load_checkpoint, _run_corpus,
    run_synthetic_recovery_gate,
)
from run_occupied_subspace_rank_selection_repair import _load_corpora_and_accounting  # noqa: E402
from statistics import stable_seed  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "occupied_subspace_membership_effect_size.json"
ANALYSIS_VERSION = "2026-09-06"

N_REPRODUCTION_GATE_SESSION_LEVELS_PER_CORPUS = 6  # x2 estimators x2 corpora = 24 individual checks, >= 10
MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP = 4  # matches the whole-session bootstrap's own minimum elsewhere in this project

DECISION_RULE_DECLARED_BEFORE_FITTING = (
    "THE EFFECT SIZE, per corpus x session x item-count-level x estimator: contrast = within_fraction - "
    "k/p, where within_fraction is the estimator-selected-rank-k basis's captured squared-norm fraction of "
    "the residual-direction axis (identical quantity the delivered pipeline computed) and k/p is the "
    "analytic chance reference for a uniformly random ambient direction against a FIXED k-dimensional "
    "subspace (derived in the module docstring; k is that cell's own selected rank, p its own ambient unit "
    "count). Within a session, levels are combined by trial-count weighting (identical convention to every "
    "other pooling in this project). Across sessions, the pooled value is the plain mean of the per-session "
    "values, with a whole-SESSION cluster bootstrap "
    f"({N_BOOT_SESSION_CLUSTER} draws, resample sessions with replacement, recompute the pooled mean each "
    "draw) supplying both a 95%-percentile confidence interval and a minimum detectable difference at 80% "
    f"power (mdd = {Z_80_POWER} * bootstrap standard error) -- never a trial-count formula. Only estimators "
    "the synthetic recovery gate (imported unchanged from the module that ran it) validates at this "
    "corpus's median ambient unit count are used; only sessions where that estimator returned a rank >= 1 "
    "(a real subspace to test membership against) contribute a session-pooled value -- a cell where the "
    "permutation eigenvalue threshold finds no eigenvalue exceeding its own permutation null is a genuine "
    "informative outcome (no detectable low-dimensional structure at all, let alone containing the axis), "
    "not folded into the contrast as a zero.\n"
    "PER-ESTIMATOR VERDICT, from the pooled contrast's cluster-bootstrap 95% interval alone (three "
    "outcomes, decided in this order): (1) if fewer than "
    f"{MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP} sessions have a detected subspace to pool -> "
    "'inconclusive_too_few_sessions_with_a_detected_subspace_to_pool'; (2) if the interval's lower bound is "
    "above 0 -> 'above_chance'; (3) if the interval's upper bound is below 0 -> 'at_or_below_chance'; "
    "(4) otherwise (interval straddles 0) -> 'inconclusive_confidence_interval_includes_zero'. An estimator "
    "not validated by the synthetic gate at this corpus's scale never reaches this rule; it is reported with "
    "its raw numbers and excluded from every branch with the reason 'excluded_failed_synthetic_gate_at_this_"
    "corpus_scale'.\n"
    "ONE BRANCH PER CORPUS, from the set of estimators that are BOTH validated by the gate AND reach an "
    "'above_chance' or 'at_or_below_chance' verdict (the 'evaluable' estimators for that corpus; estimators "
    "are never averaged or otherwise combined into a single number -- agreement or disagreement among "
    "verdicts is all that is asked): if there are no evaluable estimators -> "
    "'membership_relative_to_chance_is_inconclusive_given_available_power' (the third, inconclusive, "
    "outcome this rule always leaves open); if every evaluable estimator says 'above_chance' -> "
    "'the_axis_lies_inside_the_occupied_space_beyond_chance_expectation'; if every evaluable estimator says "
    "'at_or_below_chance' -> 'the_axis_does_not_exceed_chance_expectation_within_the_selected_subspace'; any "
    "other mix of evaluable verdicts -> 'membership_relative_to_chance_is_inconclusive_given_available_power'.\n"
    "THE TWO NULLS THE DELIVERED PIPELINE ALREADY COMPUTED are carried forward per cell but fire no branch "
    "here. The within-subspace ('harder') null is disqualified outright: its null vectors are linear "
    "combinations of the same basis that spans the tested subspace, so they lie exactly inside it and their "
    "off-fraction is zero up to floating-point round-off regardless of the data -- it cannot discriminate "
    "the axis from a null constructed to agree with it, so no comparison against it is reported as evidence "
    "of anything. The ambient null is reported as a REFERENCE distribution, not a test: it is a Monte Carlo "
    "estimate of the same analytic k/p quantity this module computes in closed form (the two are cross-"
    "checked against each other below), and any two-sided empirical test of off_fraction -- a squared-norm "
    "ratio bounded below by zero -- against that null's strictly positive centre is one-sided by "
    "construction in the interesting direction: an axis with any real concentration inside a small selected "
    "subspace will show off_fraction far BELOW the ambient null's centre near-automatically, so 'off_fraction "
    "significantly below the ambient null' is disclosed here as near-guaranteed once k << p and is not, on "
    "its own, independent evidence beyond the k/p contrast already reported.\n"
    "REPRODUCTION GATE: before any of the above, the two rank estimators are recomputed fresh (bypassing "
    "the on-disk checkpoint) for a fixed, deterministic sample of session x level cells spanning both "
    "corpora and compared to the checkpointed value to a tolerance of 1e-6; every check is reported "
    "individually and the run stops without firing any branch if any check fails."
)


# =======================================================================================================
# Whole-session cluster bootstrap: confidence interval AND minimum detectable difference from one set of
# resampling draws (never a trial-count formula).
# =======================================================================================================

def _whole_session_cluster_bootstrap(values: list[float], seed_tag: str) -> dict:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    n = int(arr.size)
    if n < MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP:
        return {"status": "not_computable", "n_sessions": n,
                "reason": f"fewer than {MIN_SESSIONS_FOR_CLUSTER_BOOTSTRAP} sessions have a detected subspace to pool"}
    rng = np.random.default_rng(stable_seed(seed_tag))
    boot_means = np.array([arr[rng.integers(0, n, size=n)].mean() for _ in range(N_BOOT_SESSION_CLUSTER)])
    se = float(np.std(boot_means, ddof=1))
    ci_low, ci_high = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))
    return {
        "status": "computed", "n_sessions": n, "n_bootstrap_draws": N_BOOT_SESSION_CLUSTER,
        "pooled_mean": float(arr.mean()), "cluster_bootstrap_ci_95pct": [ci_low, ci_high],
        "cluster_bootstrap_se": se, "z_multiplier_80pct_power": Z_80_POWER,
        "minimum_detectable_difference_at_80pct_power": Z_80_POWER * se,
    }


def _estimator_verdict(contrast_pooled: dict) -> str:
    if contrast_pooled.get("status") != "computed":
        return "inconclusive_too_few_sessions_with_a_detected_subspace_to_pool"
    ci_low, ci_high = contrast_pooled["cluster_bootstrap_ci_95pct"]
    if ci_low > 0.0:
        return "above_chance"
    if ci_high < 0.0:
        return "at_or_below_chance"
    return "inconclusive_confidence_interval_includes_zero"


def _corpus_branch(evaluable_verdicts: dict[str, str]) -> str:
    if not evaluable_verdicts:
        return "membership_relative_to_chance_is_inconclusive_given_available_power"
    verdicts = set(evaluable_verdicts.values())
    if verdicts == {"above_chance"}:
        return "the_axis_lies_inside_the_occupied_space_beyond_chance_expectation"
    if verdicts == {"at_or_below_chance"}:
        return "the_axis_does_not_exceed_chance_expectation_within_the_selected_subspace"
    return "membership_relative_to_chance_is_inconclusive_given_available_power"


# =======================================================================================================
# Per-cell extraction from the (checkpoint-backed, unchanged) per-session records `_run_corpus` returns.
# =======================================================================================================

def _extract_cell_records(per_session: list[dict], estimator_name: str) -> list[dict]:
    records = []
    for sess in per_session:
        if sess.get("status") != "computed":
            continue
        for lvl in sess.get("levels", []):
            cell = lvl.get(estimator_name, {})
            if cell.get("status") != "computed":
                continue
            harder = cell.get("harder_null", {})
            ambient = cell.get("ambient_null", {})
            if harder.get("status") != "computed":
                continue
            k = cell["best_k"]
            p = lvl["n_units"]
            chance = k / p
            analytic_vs_montecarlo_ambient_null_diff = (
                None if ambient.get("null_off_fraction_mean") is None
                else float((1.0 - chance) - ambient["null_off_fraction_mean"])
            )
            records.append({
                "session": sess["session"], "level": lvl["level"], "n_trials": lvl["n_trials"],
                "selected_rank_k": k, "ambient_unit_count_p": p,
                "within_fraction": harder["within_fraction"],
                "chance_reference_k_over_p": chance,
                "contrast_within_fraction_minus_chance_reference": harder["within_fraction"] - chance,
                "ambient_null_reference_distribution_not_a_test": {
                    "off_fraction_mean": ambient.get("null_off_fraction_mean"),
                    "off_fraction_sd": ambient.get("null_off_fraction_sd"),
                    "analytic_reference_1_minus_k_over_p": 1.0 - chance,
                    "analytic_minus_montecarlo_difference": analytic_vs_montecarlo_ambient_null_diff,
                },
                "degenerate_within_subspace_null_disqualified": {
                    "off_fraction_mean": harder.get("null_off_fraction_mean"),
                    "off_fraction_sd": harder.get("null_off_fraction_sd"),
                    "off_fraction_above_null_flag_from_the_delivered_pipeline": harder.get("off_fraction_above_null"),
                    "null_is_degenerate_by_construction": harder.get("null_is_degenerate_by_construction"),
                    "disqualified_reason": (
                        "its null vectors are coeffs @ basis.T -- linear combinations of the same basis "
                        "that spans the tested subspace -- so they lie exactly inside it and their "
                        "off-fraction is zero up to floating-point round-off regardless of the data; any "
                        "strictly positive observed off-fraction is therefore guaranteed to test as "
                        "'significant' against it. Excluded from every branch decision in this module."
                    ),
                },
            })
    return records


def _pool_session_levels(records_for_one_session: list[dict]) -> tuple[float | None, float | None]:
    wf = _trial_count_weighted([(r["n_trials"], r["within_fraction"]) for r in records_for_one_session])
    kp = _trial_count_weighted([(r["n_trials"], r["chance_reference_k_over_p"]) for r in records_for_one_session])
    return wf, kp


def _pool_estimator_for_corpus(records: list[dict], corpus_key: str, estimator_name: str) -> dict:
    by_session: dict[str, list[dict]] = {}
    for r in records:
        by_session.setdefault(r["session"], []).append(r)
    session_wf, session_kp, session_contrast, session_ids = [], [], [], []
    for session, recs in sorted(by_session.items()):
        wf, kp = _pool_session_levels(recs)
        if wf is None or kp is None:
            continue
        session_wf.append(wf)
        session_kp.append(kp)
        session_contrast.append(wf - kp)
        session_ids.append(session)
    tag = f"occupied_subspace_membership_effect_size|{corpus_key}|{estimator_name}"
    within_pooled = _whole_session_cluster_bootstrap(session_wf, f"{tag}|within_fraction")
    chance_pooled = _whole_session_cluster_bootstrap(session_kp, f"{tag}|chance_reference")
    contrast_pooled = _whole_session_cluster_bootstrap(session_contrast, f"{tag}|contrast")
    return {
        "n_sessions_with_a_detected_subspace": len(session_ids),
        "session_ids_with_a_detected_subspace": session_ids,
        "within_fraction_pooled": within_pooled,
        "chance_reference_pooled": chance_pooled,
        "contrast_pooled": contrast_pooled,
        "verdict": _estimator_verdict(contrast_pooled),
    }


# =======================================================================================================
# REPRODUCTION GATE -- independent recomputation (bypassing the on-disk checkpoint) of a fixed sample of
# per-session-level cells, compared to the checkpointed value this run otherwise reuses.
# =======================================================================================================

def run_reproduction_recomputation_gate(bundles: dict[str, list[dict]]) -> dict:
    checks = []
    for corpus_key in CORPORA:
        axis_entries = _collect_axis_entries(bundles[corpus_key], corpus_key)
        n_taken = 0
        for bundle in bundles[corpus_key]:
            if n_taken >= N_REPRODUCTION_GATE_SESSION_LEVELS_PER_CORPUS:
                break
            session = bundle["session"]
            entries = axis_entries.get(session, [])
            if not entries:
                continue
            entry = entries[0]
            n_taken += 1
            R, U, level = entry["R"], entry["U"], entry["level"]
            axis = leading_eigenvector(R)
            tag_base = f"occupied_subspace_rank_estimation|{corpus_key}|{session}|{level}"
            cached_session_record = _load_checkpoint(f"cells|{corpus_key}|{session}")
            cached_level = None
            if cached_session_record is not None:
                for cand in cached_session_record.get("levels", []):
                    if cand.get("level") == level:
                        cached_level = cand
                        break
            for est in ESTIMATOR_NAMES:
                fresh = _cell_under_estimator(U, axis, est, f"{tag_base}|{est}")
                cached = (cached_level or {}).get(est, {"status": "no_checkpoint_found"})
                status_match = fresh.get("status") == cached.get("status")
                within_diff = None
                within_pass = status_match
                if status_match and fresh.get("status") == "computed":
                    fresh_wf = fresh.get("harder_null", {}).get("within_fraction")
                    cached_wf = cached.get("harder_null", {}).get("within_fraction")
                    if fresh_wf is not None and cached_wf is not None:
                        within_diff = float(abs(fresh_wf - cached_wf))
                        within_pass = within_diff <= 1e-6
                    else:
                        within_pass = False
                checks.append({
                    "corpus": corpus_key, "session": session, "level": level, "estimator": est,
                    "fresh_status": fresh.get("status"), "cached_status": cached.get("status"),
                    "within_fraction_absolute_difference": within_diff,
                    "passed": bool(within_pass),
                })
    n_pass = sum(1 for c in checks if c["passed"])
    return {
        "tolerance_absolute_within_fraction_difference": 1e-6,
        "n_checks": len(checks), "n_passed": n_pass, "n_failed": len(checks) - n_pass,
        "all_passed": bool(n_pass == len(checks) and len(checks) >= 10),
        "checks": checks,
    }


# =======================================================================================================
# ORCHESTRATION
# =======================================================================================================

def main() -> None:
    t0 = time.time()
    root = data_root()

    output: dict = {
        "version": ANALYSIS_VERSION,
        "scope": (
            "Re-estimates, as an effect size against an analytic chance reference rather than a "
            "hypothesis-test null, whether the residual-direction axis lies inside the occupied state "
            "space, for the same two macaque corpora (single-item lateral prefrontal cortex, 11 sessions; "
            "multi-object, 41 sessions with multiple item-count levels) and the same two synthetically-"
            "gated rank estimators the prior re-estimation validated and applied, whose on-disk checkpoints "
            "are reused directly."
        ),
        "decision_rule_declared_before_fitting": DECISION_RULE_DECLARED_BEFORE_FITTING,
        "status": "running",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))

    print("loading both corpora (identical loaders to the prior re-estimation)", file=sys.stderr)
    loaded, zero_drop = _load_corpora_and_accounting(root)
    gate_result = loaded["gate_result"]
    bundles = loaded["bundles"]
    output["data_reproduction_gate"] = {"status": gate_result["status"]}
    output["zero_drop_accounting"] = zero_drop
    if gate_result["status"] != "reproduced_exactly":
        output["status"] = "void_data_reproduction_gate_did_not_reproduce"
        output["git_commit"] = git_commit(ROOT)
        output["wall_clock_s"] = time.time() - t0
        OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
        print("STOPPING: data reproduction gate did not reproduce; no new number was read", file=sys.stderr)
        return

    print("recomputing a fixed sample of per-session-level cells fresh, bypassing the checkpoint", file=sys.stderr)
    recompute_gate = run_reproduction_recomputation_gate(bundles)
    output["reproduction_recomputation_gate"] = recompute_gate
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    print(f"  reproduction recomputation gate: {recompute_gate['n_passed']}/{recompute_gate['n_checks']} passed "
          f"elapsed={time.time() - t0:.0f}s", file=sys.stderr)
    if not recompute_gate["all_passed"]:
        output["status"] = "void_reproduction_recomputation_gate_did_not_reproduce"
        output["git_commit"] = git_commit(ROOT)
        output["wall_clock_s"] = time.time() - t0
        OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
        print("STOPPING: reproduction recomputation gate did not reproduce; no new number was read", file=sys.stderr)
        return

    print("running the synthetic recovery gate (unchanged from the prior re-estimation)", file=sys.stderr)
    gate = run_synthetic_recovery_gate()
    output["synthetic_recovery_gate"] = gate
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    print(f"  gate elapsed={time.time() - t0:.0f}s", file=sys.stderr)

    output["occupied_state_space_membership_by_corpus"] = {}
    branches = {}
    for corpus_key in CORPORA:
        print(f"membership effect size: {corpus_key} ({len(bundles[corpus_key])} sessions, reusing "
              "on-disk rank-estimation checkpoints)", file=sys.stderr)
        corpus_result = _run_corpus(bundles[corpus_key], corpus_key)
        gate_status = _gate_status_for_corpus(corpus_result, gate)

        by_estimator = {}
        evaluable_verdicts = {}
        for est in ESTIMATOR_NAMES:
            validated = gate_status[est]["validated_for_this_corpus"]
            if not validated:
                by_estimator[est] = {
                    "excluded": True, "excluded_reason": "excluded_failed_synthetic_gate_at_this_corpus_scale",
                    "synthetic_gate_status": gate_status[est],
                }
                continue
            records = _extract_cell_records(corpus_result["per_session"], est)
            pooled = _pool_estimator_for_corpus(records, corpus_key, est)
            by_estimator[est] = {
                "excluded": False, "synthetic_gate_status": gate_status[est],
                "n_cells_with_a_detected_subspace": len(records),
                "cell_records": records,
                **pooled,
            }
            if pooled["verdict"] in ("above_chance", "at_or_below_chance"):
                evaluable_verdicts[est] = pooled["verdict"]
            else:
                by_estimator[est]["excluded_from_branch_reason"] = pooled["verdict"]

        branch = _corpus_branch(evaluable_verdicts)
        median_units = corpus_result["median_ambient_unit_count_across_cells"]
        output["occupied_state_space_membership_by_corpus"][corpus_key] = {
            "n_sessions_total": corpus_result["n_sessions_total"],
            "n_sessions_computed": corpus_result["n_sessions_computed"],
            "n_cells": corpus_result["n_cells"],
            "median_ambient_unit_count_across_cells": median_units,
            "by_estimator": by_estimator,
            "evaluable_estimator_verdicts": evaluable_verdicts,
            "branch": branch,
        }
        branches[corpus_key] = branch
        OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
        print(f"  {corpus_key}: branch={branch} evaluable_verdicts={evaluable_verdicts} "
              f"elapsed={time.time() - t0:.0f}s", file=sys.stderr)

    output["branch"] = branches
    output["git_commit"] = git_commit(ROOT)
    output["status"] = "complete"
    output["wall_clock_s"] = time.time() - t0
    OUTPUT_PATH.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float))
    print(json.dumps({"data_reproduction_gate": gate_result["status"],
                       "reproduction_recomputation_gate": {"n_passed": recompute_gate["n_passed"],
                                                            "n_checks": recompute_gate["n_checks"]},
                       "branch": branches, "wall_clock_s": output["wall_clock_s"]}, indent=2, default=float))
    print("FINISHED run_occupied_subspace_membership_effect_size", file=sys.stderr)


if __name__ == "__main__":
    main()
