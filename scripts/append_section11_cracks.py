#!/usr/bin/env python3
"""One-off: append the crack-register entries defined below, not
already covered by an existing entry. Run once; entries are added only if
their crack_id is not already present."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json  # noqa: E402

RESULTS = ROOT / "results"

NEW_ENTRIES = [
    {
        "crack_id": "identifiability_is_a_peaked_nonmonotone_function_of_true_lambda",
        "trigger": (
            "Does the confined-drift estimator's identified_fraction increase monotonically "
            "with true confinement strength, so that a lower identified_fraction can be read as "
            "'less confined' or 'weaker signal'?"
        ),
        "chase": (
            "Read identified_fraction directly off results/lambda_estimator_limits.json's "
            "true_confinement_present block across five planted lambda values at real corpus "
            "dimensions (200 reps each): lambda=0.5/s -> 0.135, 1.0/s -> 0.525, 1.65/s -> 0.350, "
            "2.1/s -> 0.180, 3.0/s -> 0.050."
        ),
        "resolution": (
            "Confirmed peaked, not monotone: identified_fraction rises from 0.135 at lambda=0.5 "
            "to a peak of 0.525 at lambda=1.0, then falls to 0.050 by lambda=3.0. A lower "
            "identified_fraction is therefore ambiguous between 'too slow to resolve in this "
            "window' (below the peak) and 'too fast for this bin width' (above the peak) -- "
            "exactly the two-way failure split PAPER_REPORT.tex's identifiability "
            "discussion states directionally, which this entry gives the peaked curve behind. No "
            "directional reading of identified_fraction is licensed by this estimator at any "
            "lambda value without also stating which side of the peak it is on."
        ),
        "status": "resolved_as_documented_property",
        "artifact": "results/lambda_estimator_limits.json",
    },
    {
        "crack_id": "identifiability_acceptance_region_differs_by_dataset_random_effect_level",
        "trigger": (
            "structure_identifiability_model.json models 'identified' (0/1) as the outcome with "
            "dataset as a random-intercept level. Is the same lambda value equally likely to "
            "register as 'identified' regardless of which dataset's delay window it was fit in?"
        ),
        "chase": (
            "confinement_identifiability's acceptance band is "
            "[min_time_constants/duration, max_lambda_dt/dt]. Every eligible corpus here shares "
            "dt=0.1s (giving the same upper bound, max_lambda_dt/dt = 0.25/0.1 = 2.5/s), but "
            "delay_length_s (== duration) is an exact function of dataset: 2.3s for "
            "000469/001187_000673, 3.0s for 000574."
        ),
        "resolution": (
            "Confirmed the band's lower edge differs by dataset: "
            "min_time_constants/duration = 2.0/2.3 = 0.870/s for 000469/001187_000673, versus "
            "2.0/3.0 = 0.667/s for 000574. A true lambda between 0.667/s and 0.870/s is "
            "unidentifiable in the two shorter-delay datasets but identifiable in 000574 -- the "
            "mixed model's Bernoulli outcome is therefore defined over a band that itself shifts "
            "with the random-effect level it is meant to average over, which is exactly why "
            "delay_length_s enters the model as its own fixed-effect covariate (its "
            "coefficient is the 000574-versus-rest dataset contrast, not a delay effect) "
            "rather than being left to the random intercept alone. Not a bug in the model "
            "specification, but a fact about it that a reader comparing identified_fraction "
            "across datasets without this covariate would miss."
        ),
        "status": "documented",
        "artifact": "src/drift_dynamics.py, results/structure_identifiability_model.json",
    },
    {
        "crack_id": "matched_draw_bootstrap_ci_and_exact_sign_flip_p_disagree_on_deciding_contrast",
        "trigger": (
            "results/structure_identifiability_matched_draws.json's hippocampus-vs-pre-SMA pair "
            "reports interval_excludes_zero: true (patient-bootstrap CI [-0.444, -0.063]) -- is "
            "that the same claim as 'this pair is significant'?"
        ),
        "chase": (
            "Compared interval_excludes_zero against the same row's own sign_flip_p_value "
            "(the exact permutation test, not the bootstrap interval) for this pair."
        ),
        "resolution": (
            "They disagree: interval_excludes_zero is true, but sign_flip_p_value = 0.076, which "
            "does not clear the predeclared p<0.05 bar (the minimum attainable exact p at n=8 "
            "paired patients is 0.0039, so this is not a power ceiling). "
            "results/structure_identifiability_model.json's revised predeclared_decision (Section "
            "8) now requires the exact sign-flip p specifically, not interval exclusion, which is "
            "why the deciding contrast's leg_agreement is 'disagree' rather than 'agree_significant' "
            "despite the bootstrap interval alone looking significant. interval_excludes_zero "
            "should not be read as a significance claim on its own in this or any other artifact "
            "that reports it alongside an exact test."
        ),
        "status": "resolved_by_using_the_exact_test_as_the_decision_criterion",
        "artifact": "results/structure_identifiability_matched_draws.json, results/structure_identifiability_model.json",
    },
    {
        "crack_id": "structure_registry_pools_single_unit_and_depth_lfp_estimates",
        "trigger": (
            "results/structure_registry.json's confinement_rate_lambda_pooled_across_datasets "
            "pools every dataset with an estimate for a structure, including both single-unit "
            "spike-count datasets (000469, 001187/000673, 000574) and depth-LFP high-gamma-power "
            "datasets (ds004752, ds005489, ds005557) -- these are different observables (a "
            "spike-count state-space fit versus a high-gamma-power state-space fit) sharing a "
            "lambda unit by convention, not by construction."
        ),
        "chase": (
            "Added confinement_rate_lambda_pooled_single_unit_only (pooled over single-unit "
            "datasets only) and an explicit modality_caveat string per structure to "
            "scripts/build_structure_registry.py, fired whenever a structure's pool actually "
            "mixes both modalities."
        ),
        "resolution": (
            "Partially resolved: the mixed-modality estimate is still the default "
            "confinement_rate_lambda_pooled_across_datasets key (not removed, since some "
            "structures -- e.g. hippocampus -- have real support in both modalities and dropping "
            "the LFP contribution would discard information), but it now carries an explicit "
            "caveat and a single-unit-only alternative sits beside it, so a reader comparing "
            "structures no longer has to assume modality-homogeneity that is not there. Whether "
            "the two modalities' lambda estimates are actually comparable at all (same underlying "
            "confinement process reflected in spike counts and high-gamma power) is a separate, "
            "unresolved question this entry does not claim to answer."
        ),
        "status": "partially_resolved",
        "artifact": "scripts/build_structure_registry.py, results/structure_registry.json",
    },
    {
        "crack_id": "corpora_on_disk_and_unread_before_verification",
        "trigger": (
            "This project's own earlier framing of several already-staged corpora (Panichello "
            "2024, Inagaki ALM, watters_2026) turned out to be stale or wrong when checked "
            "directly against the data on disk -- what else on disk had never been "
            "read at all?"
        ),
        "chase": (
            "scripts/stage_battery_corpora.py inventoried every corpus directory under the "
            "configured data root and cross-referenced it against config/datasets.json and "
            "results/anatomical_census.json, producing results/corpus_staging_audit.json (19 "
            "corpora) and results/corpus_fetch_plan.json (candidates not yet on disk)."
        ),
        "resolution": (
            "See results/corpus_staging_audit.json for the full inventory and per-corpus "
            "evidence. Notable corrections this staging pass produced: Panichello 2024 resolves "
            "to three monkey clusters by session-count/date grouping (now reflected in "
            "config/datasets.json's label_convention_note), not the 'not_staged' state it "
            "previously carried; inagaki_alm5's config entry already pointed at the full "
            "27GB silicon-probe corpus, not the small corrupted alm_5/zip an earlier framing assumed; "
            "DANDI 000004 (Chandravadia recognition task) has a genuine ~2.2s maintenance "
            "interval and was fetched, but is not yet integrated into this project's structure-level pipelines "
            "for a separate, documented reason (no region-label parser for its "
            "'Hemisphere Structure' electrode-location convention) -- see "
            "src/corpus_sessions.py's docstring."
        ),
        "status": "resolved",
        "artifact": "results/corpus_staging_audit.json, results/corpus_fetch_plan.json, src/corpus_sessions.py",
    },
    {
        "crack_id": "unlabelled_unspecific_leaked_into_anatomical_ordering",
        "trigger": (
            "results/structure_registry.json's across_structure_lambda_ordering_fastest_to_"
            "slowest_estimable_only included 'unspecific' and 'unlabelled' as if they were peer "
            "anatomical structures -- these are QC/coverage bookkeeping values (a channel whose "
            "label could not be resolved), not anatomical objects."
        ),
        "chase": (
            "scripts/build_structure_registry.py's region enumeration excluded 'pooled' via set "
            "arithmetic but never excluded these two labels; fixed by explicitly removing "
            "{'unlabelled', 'unspecific'} from the anatomical region set before building "
            "structures or ranking them, and retaining their own computed values (zero-drop) "
            "under a new top-level labels_excluded_as_non_anatomical key instead of deleting them."
        ),
        "resolution": (
            "Resolved: results/structure_registry.json regenerated; "
            "across_structure_lambda_ordering_fastest_to_slowest_estimable_only no longer "
            "contains either label, and both labels' own pooled estimates remain on record under "
            "labels_excluded_as_non_anatomical."
        ),
        "status": "resolved",
        "artifact": "scripts/build_structure_registry.py, results/structure_registry.json",
    },
    {
        "crack_id": "thirty_x_identifiability_gap_was_a_truncation_artifact",
        "trigger": (
            "results/lambda_estimator_limits.json's trigger field described 'a ~30x "
            "identifiability gap between hippocampus and pre-SMA at matched unit count' as the "
            "motivating question, phrased as if the gap itself were an established fact needing "
            "an explanation rather than a number needing verification."
        ),
        "chase": (
            "The 0/200-vs-50/200 figure this gap language traces to was measured after "
            "subsampling both structures to 8 units -- inside the truncation regime this "
            "project's own simulation characterises (bias +0.61 at the lower unit-count bound, "
            "-1.64 at the upper). At patient grain in DANDI 000469, with no unit-count "
            "subsampling, the same two structures show a 1.25x gap (hippocampus 4/18 = 22.2%, "
            "pre-SMA 5/18 = 27.8%), not ~30x."
        ),
        "resolution": (
            "Resolved: results/lambda_estimator_limits.json's trigger field rewritten to state "
            "both figures and name the subsampling truncation as the reason for the discrepancy, "
            "rather than repeating the ~30x figure as if it were the real effect size "
            "."
        ),
        "status": "resolved",
        "artifact": "results/lambda_estimator_limits.json",
    },
    {
        "crack_id": "dandi_000004_recognition_task_delay_epoch_and_region_parser_gap",
        "trigger": (
            "Is DANDI 000004 excluded from Sections 4-6 because it is a recognition task with no "
            "delay period (as its task family might suggest), or for some other reason?"
        ),
        "chase": (
            "Read the trial table fields directly (delay1_time, delay2_time, stim_on_time, "
            "stim_off_time) rather than assuming from the task name."
        ),
        "resolution": (
            "Not a task-design limitation: delay1_time to delay2_time is a genuine ~2.2s "
            "maintenance interval after a ~1.0s stim_on/stim_off encoding period, comparable in "
            "structure to the Sternberg corpora this project already uses. The actual blocker is "
            "that its electrode location field uses a 'Right Hippocampus' / 'Left Amygdala' "
            "hemisphere-prefix convention neither of this project's existing region-label parsers "
            "(nwb_structure_hemisphere_suffix, nwb_boran_brainnetome_hybrid) recognizes, and it is "
            "not yet registered in config/datasets.json. Both are scoped follow-up work, not "
            "yet attempted."
        ),
        "status": "open",
        "artifact": "src/corpus_sessions.py, src/spike_pipeline.py",
    },
]


def main() -> None:
    path = RESULTS / "crack_register.json"
    data = json.loads(path.read_text())
    existing_ids = {e.get("crack_id") for e in data["entries"]}
    added = 0
    for entry in NEW_ENTRIES:
        if entry["crack_id"] not in existing_ids:
            data["entries"].append(entry)
            added += 1
    path.write_text(canonical_json(data))
    print(f"added {added} new crack entries; total now {len(data['entries'])}")


if __name__ == "__main__":
    main()
