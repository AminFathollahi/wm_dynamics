#!/usr/bin/env python3
"""Item-identity decoding accuracy across the full trial timeline (baseline,
encoding, maintenance, probe, response) in DANDI 000469 load-1 trials.

Uses geometry.time_resolved_content_decoding (diagonal-only, O(T)) rather than
the full cross-temporal generalization matrix, since the full trial window has
many more timepoints than the maintenance-only analyses elsewhere in this
project. Two alignments are computed per subject: fixation-onset-aligned
(covers baseline through the median response) and response-aligned (covers
the peri-response transition directly, since response time is self-paced and
variable across trials).

Outputs: results/full_trial_content_decoding_000469.json
Updates: results/all_statistics.json — "full_trial_content_decoding_000469" key

Run:
    conda run -n wm_dynamics python scripts/run_full_trial_content_decoding_000469.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import h5py
from spike_pipeline import load_spike_times, build_psth
from geometry import time_resolved_content_decoding
from statistics import stable_seed, paired_sign_flip_test

DATA_DIR = Path("/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/000469")
RESULTS = ROOT / "results"

BIN_MS = 100
SMOOTH_MS = 200
N_PC = 8
N_PERM = 200
MIN_TRIALS_PER_CLASS = 6

FIX_PRE_S, FIX_POST_S = 0.5, 7.5      # fixation-aligned window
RESP_PRE_S, RESP_POST_S = 2.0, 2.0    # response-aligned window


def _zscore(psth: np.ndarray) -> np.ndarray:
    mu = psth.mean(axis=0, keepdims=True)
    sd = psth.std(axis=0, keepdims=True) + 1e-8
    return (psth - mu) / sd


def process_subject(subj: str) -> dict | None:
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
        t_fix = trials["timestamps_FixationCross"][:]
        t_resp = trials["timestamps_Response"][:]
        t_enc1 = trials["timestamps_Encoding1"][:]
        t_maint = trials["timestamps_Maintenance"][:]
        t_probe = trials["timestamps_Probe"][:]

    mask = loads == 1
    if mask.sum() < MIN_TRIALS_PER_CLASS * 2:
        return None
    labels = pic_id[mask]
    class_counts = np.bincount(labels - labels.min())
    if class_counts[class_counts > 0].min() < MIN_TRIALS_PER_CLASS:
        return None

    fix_onsets = t_fix[mask] - FIX_PRE_S
    psth_fix = build_psth(spike_lists, fix_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                          window_s=FIX_PRE_S + FIX_POST_S)
    psth_fix = _zscore(psth_fix)

    resp_onsets = t_resp[mask] - RESP_PRE_S
    psth_resp = build_psth(spike_lists, resp_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                           window_s=RESP_PRE_S + RESP_POST_S)
    psth_resp = _zscore(psth_resp)

    t_idx_fix = np.arange(psth_fix.shape[2])
    t_idx_resp = np.arange(psth_resp.shape[2])

    rng_fix = np.random.default_rng(stable_seed(subj + "_fulltrial_fix"))
    res_fix = time_resolved_content_decoding(
        psth_fix, labels, t_idx_fix, n_components=min(N_PC, n_units - 2),
        n_splits=min(3, class_counts[class_counts > 0].min()), n_perm=N_PERM, rng=rng_fix,
    )
    rng_resp = np.random.default_rng(stable_seed(subj + "_fulltrial_resp"))
    res_resp = time_resolved_content_decoding(
        psth_resp, labels, t_idx_resp, n_components=min(N_PC, n_units - 2),
        n_splits=min(3, class_counts[class_counts > 0].min()), n_perm=N_PERM, rng=rng_resp,
    )

    times_fix = t_idx_fix * (BIN_MS / 1000.0) - FIX_PRE_S
    times_resp = t_idx_resp * (BIN_MS / 1000.0) - RESP_PRE_S

    # Landmark times, fixation-relative, averaged across this subject's load-1 trials
    landmarks = {
        "encoding_onset": float(np.mean(t_enc1[mask] - t_fix[mask])),
        "maintenance_onset": float(np.mean(t_maint[mask] - t_fix[mask])),
        "probe_onset": float(np.mean(t_probe[mask] - t_fix[mask])),
        "response": float(np.mean(t_resp[mask] - t_fix[mask])),
    }

    late_maint_mask = (times_fix >= landmarks["probe_onset"] - 1.0) & (times_fix < landmarks["probe_onset"])
    post_response_mask = times_resp >= 0.5

    return {
        "n_trials": int(mask.sum()), "n_units": n_units, "n_classes": int(len(class_counts[class_counts > 0])),
        "times_fixation_aligned": times_fix.tolist(), "auc_fixation_aligned": res_fix["auc_per_t"].tolist(),
        "p_fixation_aligned": res_fix["p_per_t"].tolist(),
        "times_response_aligned": times_resp.tolist(), "auc_response_aligned": res_resp["auc_per_t"].tolist(),
        "p_response_aligned": res_resp["p_per_t"].tolist(),
        "landmarks": landmarks,
        "late_maintenance_auc": float(np.nanmean(res_fix["auc_per_t"][late_maint_mask])),
        "post_response_auc": float(np.nanmean(res_resp["auc_per_t"][post_response_mask])),
    }


def main():
    per_subject = {}
    for sub_n in range(1, 22):
        subj = f"sub-{sub_n}"
        print(f"Processing {subj}...")
        result = process_subject(subj)
        if result is None:
            print(f"  SKIP {subj}")
            continue
        per_subject[subj] = result
        print(f"  N={result['n_trials']}, late-maintenance AUC={result['late_maintenance_auc']:.3f}, "
              f"post-response AUC={result['post_response_auc']:.3f}")

    late_maint = np.array([v["late_maintenance_auc"] for v in per_subject.values()])
    post_resp = np.array([v["post_response_auc"] for v in per_subject.values()])
    rng = np.random.default_rng(0)
    drop_test = paired_sign_flip_test(post_resp, late_maint, n_perm=10000, alternative="less", rng=rng)
    drop_test = {k: v for k, v in drop_test.items() if k != "null"}
    print(f"\nLate-maintenance vs. post-response AUC (N={len(per_subject)} subjects): "
          f"mean(post_response - late_maintenance)={drop_test['mean_diff']:.4f} "
          f"[{drop_test['ci_lower']:.4f}, {drop_test['ci_upper']:.4f}], p={drop_test['p_value']:.4f}")

    out = {"per_subject": per_subject, "post_response_drop_test": drop_test}
    with open(RESULTS / "full_trial_content_decoding_000469.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["full_trial_content_decoding_000469"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved results/full_trial_content_decoding_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
