#!/usr/bin/env python3
"""Item-identity decoder confidence across the full trial timeline, and its
relationship to trial outcome, in DANDI 000469 — load-1 (single item) and
load-3 (three sequentially-encoded items).

For each qualifying subject and item, geometry.out_of_fold_class_confidence
gives every trial a held-out decoder's predicted probability of that trial's
own true item identity, at every timepoint across the same fixation-aligned
and response-aligned windows used in run_full_trial_content_decoding_000469.py
and run_multiitem_recall_decoding_000469.py. This per-trial, per-timepoint
confidence is then compared between correct and error trials with
statistics.temporal_cluster_permutation_auroc, which reports the effect on
the same AUC-0.5 scale used throughout this project together with
cluster-corrected significance across time.

Reuses the window-construction logic and constants directly from
run_full_trial_content_decoding_000469.py (load-1) and
run_multiitem_recall_decoding_000469.py (load-3) rather than duplicating them.

Outputs: results/decoder_confidence_timecourse_000469.json
Updates: results/all_statistics.json — "decoder_confidence_timecourse_000469" key

Run (after run_000469_pipeline.py):
    conda run -n wm_dynamics python scripts/run_decoder_confidence_timecourse_000469.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path
sys.path.insert(0, str(ROOT / "scripts"))

import h5py
from spike_pipeline import load_spike_times, build_psth, low_rate_unit_mask, FrozenPSTHTransform
from geometry import out_of_fold_class_confidence
from statistics import gated_outcome_cluster_test
from io_utils import locked_json_update

import run_full_trial_content_decoding_000469 as single_item
import run_multiitem_recall_decoding_000469 as multi_item

DATA_DIR = dataset_path("dandi_000469")
RESULTS = ROOT / "results"
N_PC = 8


def _confidence_timecourse(psth_fix, psth_resp, labels, n_units):
    n_components = min(N_PC, n_units - 2)
    conf_fix = out_of_fold_class_confidence(psth_fix, labels, np.arange(psth_fix.shape[2]),
                                            n_components=n_components, n_splits=3)
    conf_resp = out_of_fold_class_confidence(psth_resp, labels, np.arange(psth_resp.shape[2]),
                                             n_components=n_components, n_splits=3)
    return conf_fix, conf_resp


def process_load1(subj: str, rng) -> dict | None:
    nwb_path = DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
    if not nwb_path.exists():
        return None
    with h5py.File(str(nwb_path), "r") as f:
        n_units = int(f["units/id"].shape[0])
        if n_units < 15:
            return None
        spike_lists = load_spike_times(f)
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        pic_id = trials["loadsEnc1_PicIDs"][:].astype(int)
        acc = trials["response_accuracy"][:].astype(bool)
        t_fix = trials["timestamps_FixationCross"][:]
        t_resp = trials["timestamps_Response"][:]
        t_maint = trials["timestamps_Maintenance"][:]

    # Same firing-rate QC floor as run_000469_pipeline.py (Daume et al. 2024).
    rate_mask = low_rate_unit_mask(spike_lists, t_maint, single_item.MAINT_WIN)
    if rate_mask.sum() < 15:
        return None
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
    n_units = int(rate_mask.sum())

    mask = loads == 1
    if mask.sum() < single_item.MIN_TRIALS_PER_CLASS * 2:
        return None
    labels = pic_id[mask]
    if len(np.unique(labels)) < 2:
        return None

    fix_onsets = t_fix[mask] - single_item.FIX_PRE_S
    psth_fix = FrozenPSTHTransform().fit_transform(build_psth(
        spike_lists, fix_onsets, bin_ms=single_item.BIN_MS, smooth_ms=single_item.SMOOTH_MS,
        window_s=single_item.FIX_PRE_S + single_item.FIX_POST_S))
    resp_onsets = t_resp[mask] - single_item.RESP_PRE_S
    psth_resp = FrozenPSTHTransform().fit_transform(build_psth(
        spike_lists, resp_onsets, bin_ms=single_item.BIN_MS, smooth_ms=single_item.SMOOTH_MS,
        window_s=single_item.RESP_PRE_S + single_item.RESP_POST_S))

    times_fix = np.arange(psth_fix.shape[2]) * (single_item.BIN_MS / 1000.0) - single_item.FIX_PRE_S
    times_resp = np.arange(psth_resp.shape[2]) * (single_item.BIN_MS / 1000.0) - single_item.RESP_PRE_S

    conf_fix, conf_resp = _confidence_timecourse(psth_fix, psth_resp, labels, n_units)
    test_fix = gated_outcome_cluster_test(conf_fix, acc[mask], times_fix, rng=rng)
    test_resp = gated_outcome_cluster_test(conf_resp, acc[mask], times_resp, rng=rng)

    return {
        "n_trials": int(mask.sum()),
        "times_fixation_aligned": times_fix.tolist(), "times_response_aligned": times_resp.tolist(),
        "mean_confidence_fixation_aligned": np.nanmean(conf_fix, axis=0).tolist(),
        "mean_confidence_response_aligned": np.nanmean(conf_resp, axis=0).tolist(),
        "outcome_test_fixation_aligned": test_fix, "outcome_test_response_aligned": test_resp,
    }


def process_load3(subj: str, rng) -> dict | None:
    nwb_path = DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
    if not nwb_path.exists():
        return None
    with h5py.File(str(nwb_path), "r") as f:
        n_units = int(f["units/id"].shape[0])
        if n_units < 15:
            return None
        spike_lists = load_spike_times(f)
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        acc = trials["response_accuracy"][:].astype(bool)
        t_fix = trials["timestamps_FixationCross"][:]
        t_resp = trials["timestamps_Response"][:]
        t_probe = trials["timestamps_Probe"][:]
        t_maint = trials["timestamps_Maintenance"][:]
        item_labels = {name: trials[field][:].astype(int)
                       for name, (field, _) in multi_item.ITEM_FIELDS.items()}

    # Same firing-rate QC floor as run_000469_pipeline.py (Daume et al. 2024).
    rate_mask = low_rate_unit_mask(spike_lists, t_maint, multi_item.MAINT_WIN)
    if rate_mask.sum() < 15:
        return None
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
    n_units = int(rate_mask.sum())

    mask = loads == 3
    if mask.sum() < multi_item.MIN_TRIALS_PER_CLASS * 2:
        return None
    if not all(multi_item._class_counts_ok(item_labels[name][mask]) for name in multi_item.ITEM_FIELDS):
        return None

    mean_probe_rel = float(np.mean(t_probe[mask] - t_fix[mask]))
    fix_post_s = max(multi_item.FIX_POST_S, mean_probe_rel + 2.5)

    fix_onsets = t_fix[mask] - multi_item.FIX_PRE_S
    psth_fix = FrozenPSTHTransform().fit_transform(build_psth(
        spike_lists, fix_onsets, bin_ms=multi_item.BIN_MS, smooth_ms=multi_item.SMOOTH_MS,
        window_s=multi_item.FIX_PRE_S + fix_post_s))
    resp_onsets = t_resp[mask] - multi_item.RESP_PRE_S
    psth_resp = FrozenPSTHTransform().fit_transform(build_psth(
        spike_lists, resp_onsets, bin_ms=multi_item.BIN_MS, smooth_ms=multi_item.SMOOTH_MS,
        window_s=multi_item.RESP_PRE_S + multi_item.RESP_POST_S))

    times_fix = np.arange(psth_fix.shape[2]) * (multi_item.BIN_MS / 1000.0) - multi_item.FIX_PRE_S
    times_resp = np.arange(psth_resp.shape[2]) * (multi_item.BIN_MS / 1000.0) - multi_item.RESP_PRE_S

    result = {"n_trials": int(mask.sum()),
              "times_fixation_aligned": times_fix.tolist(), "times_response_aligned": times_resp.tolist()}
    for name in multi_item.ITEM_FIELDS:
        labels = item_labels[name][mask]
        conf_fix, conf_resp = _confidence_timecourse(psth_fix, psth_resp, labels, n_units)
        test_fix = gated_outcome_cluster_test(conf_fix, acc[mask], times_fix, rng=rng)
        test_resp = gated_outcome_cluster_test(conf_resp, acc[mask], times_resp, rng=rng)
        result[name] = {
            "mean_confidence_fixation_aligned": np.nanmean(conf_fix, axis=0).tolist(),
            "mean_confidence_response_aligned": np.nanmean(conf_resp, axis=0).tolist(),
            "outcome_test_fixation_aligned": test_fix, "outcome_test_response_aligned": test_resp,
        }
    return result


def main():
    rng = np.random.default_rng(0)
    out = {"load1": {}, "load3": {}}

    for sub_n in range(1, 22):
        subj = f"sub-{sub_n}"
        print(f"load1 {subj}...")
        r1 = process_load1(subj, rng)
        if r1 is not None:
            out["load1"][subj] = r1
            for win in ["fixation_aligned", "response_aligned"]:
                t = r1[f"outcome_test_{win}"]
                if t and t["significant"]:
                    print(f"  {win}: {len(t['significant'])} significant cluster(s)")

    for sub_n in range(1, 22):
        subj = f"sub-{sub_n}"
        print(f"load3 {subj}...")
        r3 = process_load3(subj, rng)
        if r3 is not None:
            out["load3"][subj] = r3
            for name in multi_item.ITEM_FIELDS:
                for win in ["fixation_aligned", "response_aligned"]:
                    t = r3[name][f"outcome_test_{win}"]
                    if t and t["significant"]:
                        print(f"  {name} {win}: {len(t['significant'])} significant cluster(s)")

    with open(RESULTS / "decoder_confidence_timecourse_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["decoder_confidence_timecourse_000469"] = out
    print("\nSaved results/decoder_confidence_timecourse_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
