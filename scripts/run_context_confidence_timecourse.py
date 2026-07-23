#!/usr/bin/env python3
"""Context (load/set-size) decoder confidence across the maintenance window,
and its relationship to trial outcome, across every dataset with both a
load/set-size label and a trial-level correct/error label: Boran iEEG, Boran
single units, and the three DANDI Rutishauser-lineage single-unit cohorts.
Miller is excluded — no behavioral accuracy field in the public release (see
Methods).

Uses geometry.out_of_fold_class_confidence on each dataset's already-computed
latent trajectories, giving every trial a held-out decoder's confidence in
its own true load/set-size label at every timepoint, then
statistics.gated_outcome_cluster_test to test this against correct/error
outcome — the identical pipeline used for the content-code version in
run_decoder_confidence_timecourse_000469.py, so the two are directly
comparable.

Outputs: results/context_confidence_timecourse.json
Updates: results/all_statistics.json — "context_confidence_timecourse" key

Run (after the per-dataset geometry pipelines have produced their npz files):
    conda run -n wm_dynamics python scripts/run_context_confidence_timecourse.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import out_of_fold_class_confidence
from statistics import stable_seed, gated_outcome_cluster_test
from io_utils import locked_json_update

RESULTS = ROOT / "results"
N_PC = 8
MIN_TRIALS_PER_OUTCOME = 8


def _run_dataset(glob_pattern: str, load_field: str, outcome_field: str, low_val, high_val,
                 name_prefix_len: int) -> dict:
    per_session = {}
    for path in sorted(RESULTS.glob(glob_pattern)):
        key = path.stem[name_prefix_len:]
        d = np.load(path, allow_pickle=True)
        if load_field not in d or outcome_field not in d:
            continue
        Z, load_val, outcome = d["Z"], d[load_field], d[outcome_field]
        mask = (load_val == low_val) | (load_val == high_val)
        if mask.sum() < 2 * MIN_TRIALS_PER_OUTCOME:
            continue
        X = Z[mask].transpose(0, 2, 1)
        context_labels = (load_val[mask] == high_val).astype(int)
        outcome_mask = outcome[mask].astype(int)
        times = np.arange(X.shape[2])

        rng = np.random.default_rng(stable_seed(key + "_context_confidence"))
        confidence = out_of_fold_class_confidence(X, context_labels, times,
                                                  n_components=min(N_PC, X.shape[1]), n_splits=5, rng=rng)
        test = gated_outcome_cluster_test(confidence, outcome_mask, times.astype(float),
                                          min_trials_per_outcome=MIN_TRIALS_PER_OUTCOME, rng=rng)
        per_session[key] = {
            "n_trials": int(mask.sum()),
            "mean_confidence": np.nanmean(confidence, axis=0).tolist(),
            "outcome_test": test,
        }
        if test and test["significant"]:
            print(f"  {key}: {len(test['significant'])} significant cluster(s)")
    return per_session


def main():
    print("Boran iEEG...")
    boran_ieeg = _run_dataset("boran_geometry_sub-*.npz", "set_sizes", "correct", 4, 8,
                              len("boran_geometry_"))
    print("Boran single units...")
    boran_units = _run_dataset("dandi000574_units_geometry_*.npz", "set_size", "correct", 4, 8,
                               len("dandi000574_units_geometry_"))
    print("DANDI 000469...")
    d469 = _run_dataset("dandi000469_geometry_sub-*.npz", "loads", "response_accuracy", 1, 3,
                        len("dandi000469_geometry_"))
    print("DANDI 001187...")
    d1187 = _run_dataset("dandi001187_geometry_sub-*.npz", "loads", "response_accuracy", 1, 3,
                         len("dandi001187_geometry_"))
    print("DANDI 000673...")
    d673 = _run_dataset("dandi000673_geometry_sub-*.npz", "loads", "response_accuracy", 1, 3,
                        len("dandi000673_geometry_"))

    out = {"boran_ieeg": boran_ieeg, "boran_units": boran_units,
           "dandi000469": d469, "dandi001187": d1187, "dandi000673": d673}

    n_tests = sum(1 for ds in out.values() for v in ds.values() if v["outcome_test"] is not None)
    n_sig = sum(1 for ds in out.values() for v in ds.values()
               if v["outcome_test"] is not None and v["outcome_test"]["significant"])
    print(f"\nTotal: {n_tests} session-level tests, {n_sig} with a significant cluster "
          f"(chance expectation at alpha=0.05: {0.05 * n_tests:.1f})")

    with open(RESULTS / "context_confidence_timecourse.json", "w") as f:
        json.dump(out, f, indent=2)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["context_confidence_timecourse"] = out
    print("\nSaved results/context_confidence_timecourse.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
