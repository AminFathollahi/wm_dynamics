"""Rebuild the maintenance-period human stimulation prediction record from its
source artifacts: a power calculation for a proposed intracranial
maintenance-period stimulation experiment, plus the displacement, predicted
accuracy-change and targeting context it is built on."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "scripts"):
    path = str(ROOT / part)
    if path not in sys.path:
        sys.path.insert(0, path)

from scipy.stats import norm  # noqa: E402
from stimulation_events import ALPHA, POWER  # noqa: E402

PRODUCING_SCRIPT = "scripts/build_maintenance_stimulation_prediction.py"

VERSION = "2026-08-28"

SCOPE = "A power calculation and prediction-input record for a proposed intracranial human maintenance-period stimulation experiment. It licenses one thing: the subject count a between-subject correlational design of the kind already run in this project's encoding-period human stimulation corpus would need to detect this project's own pre-declared meaningful effect size, given that design's own already-measured variability. It does not license a claim that this effect size will be observed, that any specific subject pool exists, or that a within-subject design's power is known -- no delivered artifact measures within-subject variability for this kind of test, because too few subjects in the existing corpus vary dose within subject to estimate it (see within_subject_alternative below). It is not a claim about non-invasive stimulation: the transcranial-delivery numbers here are upper bounds by construction and are reported only to support the conclusion that the proposed experiment is intracranial, not to support any existence, behaviour, or targeting claim."

ALREADY_PRODUCED_READING = "Encoding-period stimulation in the two open human corpora produced a real, artifact-controlled displacement, but at roughly one-tenth to one-twelfth of the magnitude the artifact's own pre-declared threshold calls meaningful; this is why the prediction below is stated against the threshold, not against what has already been achieved."

DIRECTION = "decrease"

DIRECTION_REASONING = "Two delivered signs agree. (1) In the one real, unstimulated maintenance delay this project measures at single-unit resolution, higher deviation from the session's own reference state direction correlates positively with trial error (results/rate_free_state_geometry_behavior_link.json, key path sign_interpretation: deviation correlates with error positively). (2) The encoding-period human stimulation displacement is positive on average in the same normalised units (results/human_stimulation_component_response.json, block_b pooled mean values above are positive in both channel conditions), i.e. stimulation pushes the component in the same direction associated with worse behaviour, not the opposite one. A maintenance-period displacement of the same sign is therefore predicted to move accuracy downward, not upward."

REASONING_FOR_MAGNITUDE = "No delivered artifact measures the accuracy change produced by a maintenance-period displacement, because no open corpus stimulates inside an isolated maintenance delay (this is the structural absence the prediction is built to address). In the absence of a measured dose-response mapping, the spike-count-matched decile contrast -- the accuracy difference already measured, in the same rate-free direction, between the most- and least-deviant tenth of naturally occurring trials in a real, unstimulated maintenance delay -- is used as the predicted order of magnitude for an intervention that reaches the pre-declared meaningful displacement threshold. This is an explicit scaling assumption (a threshold-scale induced displacement is assumed to be comparable in effect to the naturally occurring top-versus-bottom-decile spread of the same observable), not a delivered dose-response estimate, and it is reported as such."

DESIGN = "A subject-clustered correlation between induced displacement and induced behaviour change (the mediation test structure already run at encoding), relocated to an intracranial maintenance-period stimulation experiment."

ASSUMED_EFFECT_SIZE_REASONING = "Reuses this project's own pre-declared meaningful-effect threshold (0.14 r units) unchanged, the same convention already applied to every other arm of the human stimulation corpus (results/human_stimulation_component_response.json block_c and block_a both carry the identical field), rather than assuming a larger or smaller effect no delivered maintenance-period measurement supports."

RATIO_FORMULA = "n_required = n_current * (mdd_current / assumed_effect_size_r) ** 2"

RATIO_NOTE = "This is the scaling already used, on the same inputs, in the manuscript's own acquisition-requirements accounting for this mediation block; reproduced here independently and serialised."

FISHER_FORMULA = "n_required = 3 + (z_factor / atanh(assumed_effect_size_r)) ** 2"

FISHER_NOTE = "Independent exact formula for the same quantity; agrees with the ratio-of-squares scaling to within 5%."

WITHIN_SUBJECT_REASONING = "Invasive human stimulation cohorts of this project's own size (47-50 subjects across both encoding-period corpora combined) are already among the largest that exist; a between-subject design at the required count is not a realistic acquisition target. The existing corpus's own attempt at a within-subject dose contrast could not be computed for the opposite reason: only a handful of subjects vary stimulation amplitude within the same electrode pair."

TARGETING_DESCRIPTION = "The alignment-based targeting heuristic already tested in the encoding-period human corpus: absolute cosine alignment between the stimulation input direction and the leading eigenvector of the session's own fitted dynamics, against the observed displacement. Underpowered against this project's own reference, not a powered null; reported here only as context for what site-selection evidence does and does not yet exist for the proposed experiment."

INTRACRANIAL_DESCRIPTION = "Every anatomical region matched within the pre-declared 10 mm radius between a human intracranial site and a transcranial-corpus electrode requires a transcranially delivered current beyond the pre-declared 4 mA deliverable-current threshold to reach the field the delivered intracranial dose actually produced. These shortfall factors are upper bounds by construction (two independent biases in the estimate both push required current upward, never downward); no region-level factor, and no comparison across regions, is drawn from this artifact here."

UPPER_BOUND_CAVEAT = "Every required-transcranial-current and shortfall-factor number in this artifact is an upper bound, not a point estimate, and two independent biases both push it upward -- neither pushes it down. (1) The induced field is a finite difference between neighbouring transcranial-corpus contacts (see transcranial_corpus_zero_drop.neighbour_separation_mm_distribution for the actual millimetre scale this was smoothed over); a finite difference over a multi-millimetre separation is biased low relative to the true local field, and a low measured field in the denominator of required_current = reference_field / measured_field inflates the required current. (2) The reference field is evaluated at the midpoint of the stimulating bipolar pair -- close to the peak of the direct-stimulation field -- and is compared against that same coarsely-sampled, smoothed transcranial estimate; comparing a peak-like numerator against a smoothed-low denominator inflates the ratio a second time, in the same direction as bias (1). Both biases inflate every shortfall factor and required-current number below; neither deflates either. The verdict that every matched region is beyond the declared threshold is unchanged under an order-of-magnitude correction for these biases in either direction -- the shortfall factors range from 82-fold to over 18,000-fold, far larger than one order of magnitude of correction could close."


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((results_dir / name).read_text())


def build(results_dir: Path) -> dict:
    stim = _load(results_dir, "human_stimulation_component_response.json")
    effect = _load(results_dir, "component_effect_size_and_anatomy.json")
    geometry = _load(results_dir, "rate_free_state_geometry_behavior_link.json")
    transcranial = _load(results_dir, "transcranial_current_requirement.json")

    block_b = stim["block_b"]
    shank_excluded = block_b["pooled_normalised_displacement_by_channel_condition"]["excluding_stimulated_shank"]
    full_channels = block_b["pooled_normalised_displacement_by_channel_condition"]["full_channel_set"]

    decile = effect["block_a"]["macaque_lPFC_single_item"]["pooled"]["spike_count_matched_decile_contrast"]
    decile_unmatched = effect["block_a"]["macaque_lPFC_single_item"]["pooled"]["decile_contrast"]
    sign_convention = effect["sign_convention"]["macaque_lPFC_single_item"]

    mediation_mdd = stim["block_c"]["mediation"]["mdd"]
    assumed_effect_size_r = stim["block_c"]["meaningful_effect_threshold_r_units"]

    z_alpha_two_sided = float(norm.ppf(1.0 - ALPHA / 2.0))
    z_power = float(norm.ppf(POWER))
    z_factor = z_alpha_two_sided + z_power

    ratio_raw = mediation_mdd["n_subjects"] * (mediation_mdd["mdd"] / assumed_effect_size_r) ** 2
    fisher_raw = 3.0 + (z_factor / math.atanh(assumed_effect_size_r)) ** 2

    within_subject = stim["block_d"]["within_subject"]

    targeting = stim["block_f"]["correlation_displacement_vs_alignment_to_vstar"]

    site_zero_drop = transcranial["human_intracranial_site_zero_drop"]
    regions = transcranial["required_transcranial_current_by_region"]
    n_regions_beyond_threshold = sum(
        1 for region in regions.values() if region["verdict"] == "beyond_the_declared_threshold"
    )

    return {
        "version": VERSION,
        "scope": SCOPE,
        "displacement_magnitude": {
            "required_for_prediction_normalised_units": block_b["meaningful_effect_threshold_normalised_displacement"],
            "required_for_prediction_source": {
                "artifact": "results/human_stimulation_component_response.json",
                "key_path": "block_b.meaningful_effect_threshold_normalised_displacement",
            },
            "required_for_prediction_source_definition": {
                "artifact": "results/human_stimulation_component_response.json",
                "key_path": "block_b.meaningful_effect_threshold_source",
                "value": block_b["meaningful_effect_threshold_source"],
            },
            "already_produced_at_encoding_context_only": {
                "shank_excluded_channel_condition": {
                    "value": shank_excluded["mean_value"],
                    "ci_lower": shank_excluded["ci_lower"],
                    "ci_upper": shank_excluded["ci_upper"],
                    "p_value": shank_excluded["p_value"],
                    "n_sessions": shank_excluded["n_sessions"],
                    "n_subjects": shank_excluded["n_subjects"],
                    "source": {
                        "artifact": "results/human_stimulation_component_response.json",
                        "key_path": "block_b.pooled_normalised_displacement_by_channel_condition.excluding_stimulated_shank",
                    },
                },
                "full_channel_set_condition": {
                    "value": full_channels["mean_value"],
                    "ci_lower": full_channels["ci_lower"],
                    "ci_upper": full_channels["ci_upper"],
                    "p_value": full_channels["p_value"],
                    "n_sessions": full_channels["n_sessions"],
                    "n_subjects": full_channels["n_subjects"],
                    "source": {
                        "artifact": "results/human_stimulation_component_response.json",
                        "key_path": "block_b.pooled_normalised_displacement_by_channel_condition.full_channel_set",
                    },
                },
                "reading": ALREADY_PRODUCED_READING,
            },
        },
        "predicted_accuracy_change": {
            "direction": DIRECTION,
            "direction_reasoning": DIRECTION_REASONING,
            "magnitude_on_matched_decile_contrast_scale": decile["mean_value"],
            "magnitude_source": {
                "artifact": "results/component_effect_size_and_anatomy.json",
                "key_path": "block_a.macaque_lPFC_single_item.pooled.spike_count_matched_decile_contrast.mean_value",
                "n_sessions": decile["n_sessions"],
                "p_value": decile["p_value"],
            },
            "unmatched_value_for_comparison": {
                "value": decile_unmatched["mean_value"],
                "n_sessions": decile_unmatched["n_sessions"],
                "source": {
                    "artifact": "results/component_effect_size_and_anatomy.json",
                    "key_path": "block_a.macaque_lPFC_single_item.pooled.decile_contrast.mean_value",
                },
            },
            "reasoning_for_using_this_as_the_predicted_size": REASONING_FOR_MAGNITUDE,
            "worse_behaviour_sign_convention_source": {
                "artifact": "results/component_effect_size_and_anatomy.json",
                "key_path": "sign_convention.macaque_lPFC_single_item",
                "value": sign_convention,
            },
        },
        "subject_count_power_calculation": {
            "design": DESIGN,
            "assumed_effect_size_r": assumed_effect_size_r,
            "assumed_effect_size_reasoning": ASSUMED_EFFECT_SIZE_REASONING,
            "alpha": ALPHA,
            "power": POWER,
            "z_alpha_two_sided": z_alpha_two_sided,
            "z_power": z_power,
            "z_factor": z_factor,
            "input_minimum_detectable_difference": {
                "value": mediation_mdd["mdd"],
                "n_subjects": mediation_mdd["n_subjects"],
                "source": {
                    "artifact": "results/human_stimulation_component_response.json",
                    "key_path": "block_c.mediation.mdd.mdd",
                },
            },
            "n_subjects_required_ratio_of_squares_scaling": {
                "formula": RATIO_FORMULA,
                "raw_value": ratio_raw,
                "n_subjects_required": int(math.ceil(ratio_raw)),
                "note": RATIO_NOTE,
            },
            "n_subjects_required_exact_fisher_z_cross_check": {
                "formula": FISHER_FORMULA,
                "raw_value": fisher_raw,
                "n_subjects_required": int(math.ceil(fisher_raw)),
                "note": FISHER_NOTE,
            },
            "quoted_in_manuscript": int(math.ceil(ratio_raw)),
            "why_within_subject_is_the_alternative_rather_than_more_subjects": {
                "reasoning": WITHIN_SUBJECT_REASONING,
                "n_subjects_with_within_subject_dose_variation": stim["block_d"]["n_subjects_with_within_subject_dose_variation"],
                "within_subject_dose_test_status": within_subject["status"],
                "within_subject_dose_test_reason": within_subject["reason"],
                "source": {
                    "artifact": "results/human_stimulation_component_response.json",
                    "key_path": "block_d.n_subjects_with_within_subject_dose_variation / block_d.within_subject",
                },
            },
        },
        "targeting_heuristic_context": {
            "description": TARGETING_DESCRIPTION,
            "r": targeting["r"],
            "n_subjects": targeting["n_subjects"],
            "n_sessions": targeting["n_sessions"],
            "minimum_detectable_difference": targeting["mdd"]["mdd"],
            "source": {
                "artifact": "results/human_stimulation_component_response.json",
                "key_path": "block_f.correlation_displacement_vs_alignment_to_vstar",
            },
        },
        "why_the_proposed_experiment_is_intracranial": {
            "description": INTRACRANIAL_DESCRIPTION,
            "n_matched_sites_within_radius": site_zero_drop["n_matched_within_radius"],
            "n_usable_sites_total": site_zero_drop["usable_for_matching"],
            "n_regions_all_beyond_threshold": n_regions_beyond_threshold,
            "deliverable_current_threshold_ma": transcranial["predeclared_constants"]["unreachable_current_threshold_ma"],
            "upper_bound_caveat": UPPER_BOUND_CAVEAT,
            "source": {
                "artifact": "results/transcranial_current_requirement.json",
                "key_path": "human_intracranial_site_zero_drop / required_transcranial_current_by_region / predeclared_constants / upper_bound_caveat",
            },
        },
        "producing_script": PRODUCING_SCRIPT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "maintenance_stimulation_prediction.json")
    args = parser.parse_args()
    record = build(args.results_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
