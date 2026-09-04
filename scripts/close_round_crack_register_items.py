#!/usr/bin/env python3
"""Crack-register bookkeeping that belongs to no single
analysis script: re-opening a crack marked resolved while its own artifact
is incomplete, and recording the matched-unit-count design as a dead end so
no future run repeats it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    crack_path = RESULTS / "crack_register.json"
    cracks = json.loads(crack_path.read_text())
    matched = json.loads((RESULTS / "region_unit_count_matched_lambda.json").read_text())

    for entry in cracks["entries"]:
        if entry.get("crack_id") == "lambda_ordering_underpowered_single_patient_leakage":
            entry.update({
                "status": "reopened_superseded_by_structure_identifiability_model",
                "trigger": (
                    "Marked 'resolved' while its own artifact "
                    "(results/region_unit_count_matched_lambda.json) is run_complete: false -- "
                    "still true: n_patients_completed="
                    f"{matched.get('n_patients_completed')}, run_complete={matched.get('run_complete')}."
                ),
                "chase": (
                    "Not re-run: run_unit_count_matched_sensitivity.py is a dead end (see "
                    "matched_count_design_cannot_identify_paired_lambda_difference below) and is "
                    "not restarted or extended. Replaced by "
                    "results/structure_identifiability_model.json (identifiability itself as the "
                    "outcome, full denominator) and "
                    "results/structure_identifiability_matched_draws.json (jointly matched on "
                    "unit count AND rate, fraction_identified as the outcome)."
                ),
                "resolution": (
                    "Superseded, not resolved. The replacement analyses answer a differently-"
                    "framed and better-controlled question and both support "
                    "structure_dissociation_supported; this crack's original artifact remains "
                    "unfinished and is not cited as evidence for or against that verdict."
                ),
                "artifact": "results/region_unit_count_matched_lambda.json (superseded by results/structure_identifiability_model.json)",
            })
            break

    if not any(e.get("crack_id") == "displacement_snr_no_common_scale_across_structures" for e in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "displacement_snr_no_common_scale_across_structures",
            "trigger": (
                "results/structure_control_observables.json reports displacement_snr_by_delta with "
                "Delta on a grid of per-structure PCA latent state units, and stationary variance "
                "ranges 0.105-0.119 across structures: the latent is not whitened, so one state unit "
                "is not the same physical displacement in amygdala as in dACC, and the cross-"
                "structure SNR column has no common scale to compare on."
            ),
            "chase": (
                "results/fidelity_controllability_map.json compares structures only on scale-free "
                "quantities: the ratio D/lambda (stationary variance, dimensionless in state units "
                "that cancel) and the ms-valued control bandwidth 1/lambda (a common physical scale, "
                "time). displacement_snr_by_delta is not recomputed or normalised."
            ),
            "resolution": (
                "Resolved by scope restriction, not by fixing the scale: "
                "the displacement_snr_by_delta column in structure_control_observables.json is "
                "superseded and excluded from every cross-structure comparison; only D/lambda and "
                "1/lambda are reported across structures."
            ),
            "status": "resolved_scope_restricted_to_scale_free_quantities",
            "artifact": "results/fidelity_controllability_map.json",
        })

    if not any(e.get("crack_id") == "structure_registry_pooled_non_identified_singletons" for e in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "structure_registry_pooled_non_identified_singletons",
            "trigger": (
                "scripts/build_structure_registry.py let a structure enter "
                "across_structure_lambda_ordering_fastest_to_slowest below the minimum patient "
                "threshold: the pooling gate compared the raw collected count (n_total, summed "
                "across per-dataset draws before any identifiability filtering) against the "
                "threshold instead of the pooled bootstrap's own finite n_patients, and a second "
                "'pooled' key leaked back into all_regions through operator-precedence-mismatched "
                "set arithmetic (`a | b | c | d - {'pooled'}` only strips the key from d). vmPFC "
                "entered the ordering at n=2, pooled from two n=1 non_identified singletons."
            ),
            "chase": (
                "Fixed at the shared pooling helper in build_structure_registry.py, not per call "
                "site: the gate now reads pooled_lambda.get('n_patients', 0) only when "
                "pooled_lambda['status'] == 'estimable', and the 'pooled' key exclusion is grouped "
                "as (union of all four source sets) - {'pooled'}. Regression tests added: "
                "test_pooled_gate_uses_the_bootstraps_finite_count_not_raw_collected_n and "
                "test_pooled_key_excluded_even_when_present_in_every_source_set."
            ),
            "resolution": (
                "Resolved. results/structure_registry.json regenerated; vmPFC no longer appears in "
                "across_structure_lambda_ordering_fastest_to_slowest."
            ),
            "status": "resolved",
            "artifact": "scripts/build_structure_registry.py",
        })

    if not any(e.get("crack_id") == "predeclared_decision_two_branch_gate_hole" for e in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "predeclared_decision_two_branch_gate_hole",
            "trigger": (
                "House-rule audit of every predeclared_decision/"
                "gate_decision rule in results/*.json and its implementing code, looking for a rule "
                "that silently folds estimator non-identification into a refuted/null branch instead "
                "of naming it. scripts/run_unit_count_matched_sensitivity.py's gate_decision has "
                "exactly two outcomes once the pooled matched-patient group is itself estimable "
                "(n>=2 patients with a finite matched-draw median): 'supported' or "
                "'identifiability_artifact'. A wide or zero-crossing patient-bootstrap interval is "
                "reported as 'identifiability_artifact' regardless of whether it reflects a genuine "
                "null or a low per-patient fraction_identified (tracked separately in "
                "identification_fraction_by_matched_unit_count, the field's current name -- renamed "
                "from identification_fraction_by_unit_count_comments_txt_4_4 -- but not fed back into "
                "the top-level verdict)."
            ),
            "chase": (
                "Not re-run: run_unit_count_matched_sensitivity.py is itself a recorded dead end "
                "(see matched_count_design_cannot_identify_paired_lambda_difference) and its output "
                "is already superseded by results/structure_identifiability_model.json and "
                "results/structure_identifiability_matched_draws.json, neither of which has this "
                "hole -- both treat non-identification as a distinct outcome. Every other "
                "predeclared_decision/gate_decision definition found this audit (fidelity_"
                "controllability_map.json, intrinsic_timescale_vs_confinement.json, "
                "structure_identifiability_model.json, tau_estimator_calibration.json, "
                "rotation_adjudication.json via run_rotation_adjudication.py, "
                "switching_adjudication.json via run_switching_adjudication.py, the rotation gate "
                "in run_vstar_fit_selection_factorial.py) already names non-identification as its "
                "own branch and was not modified."
            ),
            "resolution": (
                "Recorded, not fixed: the hole is in a superseded script whose output is not cited "
                "for the structure-dissociation verdict, so patching it would not change any "
                "reported result. Left as a named defect for anyone who reruns that script directly."
            ),
            "status": "open_in_superseded_script",
            "artifact": "scripts/run_unit_count_matched_sensitivity.py",
        })

    if not any(e.get("crack_id") == "strict_json_writers_repo_wide_beyond_named_five" for e in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "strict_json_writers_repo_wide_beyond_named_five",
            "trigger": (
                "An earlier audit named five artifacts with literal NaN/Infinity tokens: "
                "causal_benchmark.json, forest_syntheses.json, rl_policy_arm.json, "
                "all_statistics.json, targeting_benchmark_boran.json. Auditing every json.dump call "
                "site across scripts/ (not just the five) found roughly 50 more direct writers, in "
                "scripts unrelated to the five named artifacts above, with the same latent defect: a bare "
                "json.dump(...) or a json.dump(..., default=lambda o: float(o) if isinstance(o, "
                "np.floating) else o) call, neither of which converts a non-finite float to null "
                "before the encoder sees it."
            ),
            "chase": (
                "Fixed at the writing site for all direct writers of the five named artifacts "
                "(io_utils.locked_json_update for every caller that already used it, plus every "
                "direct json.dump call for all_statistics.json, causal_benchmark.json, "
                "forest_syntheses.json, rl_policy_arm.json, and targeting_benchmark_boran.json), "
                "then sanitized the five on-disk files in place so they parse under strict JSON now. "
                "Added tests/test_results_strict_json.py, parametrized over every file currently in "
                "results/, to catch regressions and flag any future artifact with the same defect."
            ),
            "resolution": (
                "Declared reduced scope: the ~50 other direct writers are not "
                "touched here -- fixing every one is a mechanical but unbounded-effort sweep "
                "outside this fix's scope, and the new results/ test already catches any "
                "of them that currently emits a non-finite value on disk. It will fail loudly the "
                "next time one of those scripts is rerun and writes a NaN, at which point it should "
                "be fixed the same way (wrap the dumped object in provenance._json_safe, drop any "
                "default=lambda numpy-only shim, and pass allow_nan=False)."
            ),
            "status": "declared_reduced_scope",
            "artifact": "tests/test_results_strict_json.py",
        })

    if not any(e.get("crack_id") == "matched_count_design_cannot_identify_paired_lambda_difference" for e in cracks["entries"]):
        cracks["entries"].append({
            "crack_id": "matched_count_design_cannot_identify_paired_lambda_difference",
            "trigger": (
                "Original design: match structures on unit count only, then compare the paired "
                "lambda difference conditional on both arms being identified. At the matched "
                "count, hippocampus lambda was identifiable in 0/200 draws against pre-SMA's "
                "50/200 in one patient, and 2/200 against 69/200 in a second -- a paired "
                "difference with (near-)zero support in the arm that matters."
            ),
            "chase": (
                "Reframed identifiability itself as the outcome rather than a nuisance to "
                "condition away. results/lambda_estimator_limits.json's selection-bias analysis "
                "shows WHY conditioning on identifiability is unsafe even where it succeeds: "
                "lambda-hat among identified draws is biased by up to +0.61 near the "
                "identifiable window's lower boundary and -1.64 near its "
                "upper boundary -- exactly the truncation regime a matched-count-only design "
                "walks into for a structure with sparse signal. Any lambda difference this design "
                "eventually reports, at any n, inherits that selection bias uncorrected."
            ),
            "resolution": (
                "Dead end, recorded so no future round repeats it: the design cannot separate "
                "'this structure has no confined-Gaussian signature' from 'this estimator is "
                "conditionally biased when it happens to succeed on a sparse signal' -- both "
                "produce the same observable (few identified draws, and among those, a "
                "boundary-shifted lambda). results/structure_identifiability_model.json and "
                "results/structure_identifiability_matched_draws.json replace it: identifiability "
                "(not lambda conditional on identifiability) is the outcome, on the full "
                "denominator, matched jointly on unit count AND rate."
            ),
            "status": "recorded_dead_end",
            "artifact": "scripts/run_unit_count_matched_sensitivity.py",
        })

    crack_path.write_text(canonical_json(cracks))
    print(json.dumps({"n_entries": len(cracks["entries"])}, indent=2))


if __name__ == "__main__":
    main()
