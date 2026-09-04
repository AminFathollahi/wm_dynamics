#!/usr/bin/env python3
"""Item-identity decoding accuracy across the full trial timeline for each of
the three sequentially-encoded items in DANDI 000469 load-3 trials.

loadsEnc1_PicIDs, loadsEnc2_PicIDs, and loadsEnc3_PicIDs each identify a
5-class, repeated-picture identity (~8-10 trials/class within load-3 trials),
encoded in sequence (encoding 1, then 2, then 3) before a single joint
maintenance period and one probe/response. Three separate decoders (one per
encoded item) are run across the same full-trial timeline (fixation onset
through response), using geometry.time_resolved_content_decoding, to test
whether each item's decodability rises at its own encoding onset and whether
all three drop together after the shared probe/response.

Outputs: results/multiitem_recall_decoding_000469.json
Updates: results/all_statistics.json — "multiitem_recall_decoding_000469" key

Run:
    conda run -n wm_dynamics python scripts/run_multiitem_recall_decoding_000469.py
"""
import sys, json, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
from spike_pipeline import load_spike_times, build_psth, low_rate_unit_mask, FrozenPSTHTransform
from geometry import time_resolved_content_decoding
from statistics import stable_seed, paired_sign_flip_test
from provenance import _json_safe

DATA_DIR = dataset_path("dandi_000469")
RESULTS = ROOT / "results"

BIN_MS = 100
SMOOTH_MS = 200
N_PC = 8
N_PERM = 200
MIN_TRIALS_PER_CLASS = 6
MIN_UNITS = 15
MAINT_WIN = 2.3  # matches run_000469_pipeline.py's true-delay-bounded window

FIX_PRE_S, FIX_POST_S = 0.5, 9.5      # fixation-aligned window, covers load-3's longer trial
RESP_PRE_S, RESP_POST_S = 2.0, 2.0    # response-aligned window

ITEM_FIELDS = {
    "item1": ("loadsEnc1_PicIDs", "timestamps_Encoding1"),
    "item2": ("loadsEnc2_PicIDs", "timestamps_Encoding2"),
    "item3": ("loadsEnc3_PicIDs", "timestamps_Encoding3"),
}


def _class_counts_ok(labels: np.ndarray) -> bool:
    counts = np.bincount(labels - labels.min())
    return counts[counts > 0].min() >= MIN_TRIALS_PER_CLASS and len(counts[counts > 0]) >= 2


def process_subject(subj: str) -> dict | None:
    nwb_path = DATA_DIR / subj / f"{subj}_ses-2_ecephys+image.nwb"
    if not nwb_path.exists():
        return None
    with h5py.File(str(nwb_path), "r") as f:
        n_units = int(f["units/id"].shape[0])
        if n_units < MIN_UNITS:
            return None
        spike_lists = load_spike_times(f)
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        t_fix = trials["timestamps_FixationCross"][:]
        t_resp = trials["timestamps_Response"][:]
        t_maint = trials["timestamps_Maintenance"][:]
        t_probe = trials["timestamps_Probe"][:]
        item_labels = {name: trials[field][:].astype(int) for name, (field, _) in ITEM_FIELDS.items()}
        item_onsets = {name: trials[onset_field][:] for name, (_, onset_field) in ITEM_FIELDS.items()}

    # Same firing-rate QC floor as run_000469_pipeline.py (Daume et al. 2024),
    # applied here too since this script re-reads raw spike trains directly
    # rather than reusing that script's already-QC'd geometry output.
    rate_mask = low_rate_unit_mask(spike_lists, t_maint, MAINT_WIN)
    if rate_mask.sum() < MIN_UNITS:
        return None
    spike_lists = [spk for spk, keep in zip(spike_lists, rate_mask) if keep]
    n_units = int(rate_mask.sum())

    mask = loads == 3
    if mask.sum() < MIN_TRIALS_PER_CLASS * 2:
        return None
    if not all(_class_counts_ok(item_labels[name][mask]) for name in ITEM_FIELDS):
        return None

    # Trial timing scale varies substantially across subjects (mean probe onset
    # ranges from ~6.8s to ~11.5s post-fixation), so the fixation-aligned
    # window is sized per subject to comfortably cover this subject's own
    # probe onset, rather than a fixed global duration.
    mean_probe_rel = float(np.mean(t_probe[mask] - t_fix[mask]))
    fix_post_s = max(FIX_POST_S, mean_probe_rel + 2.5)

    fix_onsets = t_fix[mask] - FIX_PRE_S
    psth_fix = FrozenPSTHTransform().fit_transform(
        build_psth(spike_lists, fix_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                   window_s=FIX_PRE_S + fix_post_s))
    resp_onsets = t_resp[mask] - RESP_PRE_S
    psth_resp = FrozenPSTHTransform().fit_transform(
        build_psth(spike_lists, resp_onsets, bin_ms=BIN_MS, smooth_ms=SMOOTH_MS,
                   window_s=RESP_PRE_S + RESP_POST_S))

    times_fix = np.arange(psth_fix.shape[2]) * (BIN_MS / 1000.0) - FIX_PRE_S
    times_resp = np.arange(psth_resp.shape[2]) * (BIN_MS / 1000.0) - RESP_PRE_S

    landmarks = {
        "encoding1_onset": float(np.mean(item_onsets["item1"][mask] - t_fix[mask])),
        "encoding2_onset": float(np.mean(item_onsets["item2"][mask] - t_fix[mask])),
        "encoding3_onset": float(np.mean(item_onsets["item3"][mask] - t_fix[mask])),
        "maintenance_onset": float(np.mean(t_maint[mask] - t_fix[mask])),
        "probe_onset": float(np.mean(t_probe[mask] - t_fix[mask])),
        "response": float(np.mean(t_resp[mask] - t_fix[mask])),
    }

    result = {"n_trials": int(mask.sum()), "n_units": n_units, "landmarks": landmarks,
              "times_fixation_aligned": times_fix.tolist(), "times_response_aligned": times_resp.tolist()}

    for name in ITEM_FIELDS:
        labels = item_labels[name][mask]
        n_classes = len(np.unique(labels))
        rng_fix = np.random.default_rng(stable_seed(f"{subj}_{name}_multiitem_fix"))
        res_fix = time_resolved_content_decoding(
            psth_fix, labels, np.arange(psth_fix.shape[2]), n_components=min(N_PC, n_units - 2),
            n_splits=3, n_perm=N_PERM, rng=rng_fix,
        )
        rng_resp = np.random.default_rng(stable_seed(f"{subj}_{name}_multiitem_resp"))
        res_resp = time_resolved_content_decoding(
            psth_resp, labels, np.arange(psth_resp.shape[2]), n_components=min(N_PC, n_units - 2),
            n_splits=3, n_perm=N_PERM, rng=rng_resp,
        )
        own_encoding_onset = landmarks[f"{name.replace('item', 'encoding')}_onset"]
        pre_encoding_mask = times_fix < own_encoding_onset
        late_maint_mask = (times_fix >= landmarks["probe_onset"] - 1.0) & (times_fix < landmarks["probe_onset"])
        post_response_mask = times_resp >= 0.5

        result[name] = {
            "n_classes": n_classes,
            "auc_fixation_aligned": res_fix["auc_per_t"].tolist(),
            "auc_response_aligned": res_resp["auc_per_t"].tolist(),
            "pre_own_encoding_auc": float(np.nanmean(res_fix["auc_per_t"][pre_encoding_mask])),
            "late_maintenance_auc": float(np.nanmean(res_fix["auc_per_t"][late_maint_mask])),
            "post_response_auc": float(np.nanmean(res_resp["auc_per_t"][post_response_mask])),
        }

    return result


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
        for name in ITEM_FIELDS:
            r = result[name]
            print(f"  {name}: pre-encoding={r['pre_own_encoding_auc']:.3f}, "
                  f"late-maintenance={r['late_maintenance_auc']:.3f}, "
                  f"post-response={r['post_response_auc']:.3f}")

    rng = np.random.default_rng(0)
    tests = {}
    for name in ITEM_FIELDS:
        late_maint = np.array([v[name]["late_maintenance_auc"] for v in per_subject.values()])
        post_resp = np.array([v[name]["post_response_auc"] for v in per_subject.values()])
        pre_enc = np.array([v[name]["pre_own_encoding_auc"] for v in per_subject.values()])
        valid = np.isfinite(late_maint) & np.isfinite(post_resp) & np.isfinite(pre_enc)
        if valid.sum() < len(late_maint):
            print(f"  {name}: dropping {(~valid).sum()} subject(s) with a non-finite window average")
        late_maint, post_resp, pre_enc = late_maint[valid], post_resp[valid], pre_enc[valid]
        drop_test = paired_sign_flip_test(post_resp, late_maint, n_perm=10000, alternative="less", rng=rng)
        rise_test = paired_sign_flip_test(late_maint, pre_enc, n_perm=10000, alternative="greater", rng=rng)
        tests[name] = {
            "rise_after_own_encoding": {k: v for k, v in rise_test.items() if k != "null"},
            "drop_after_response": {k: v for k, v in drop_test.items() if k != "null"},
        }
        print(f"\n{name} (N={valid.sum()} subjects): "
              f"rise (late-maint vs pre-encoding) p={rise_test['p_value']:.4f}, "
              f"drop (post-response vs late-maint) p={drop_test['p_value']:.4f}")

    out = {"per_subject": per_subject, "tests": tests}
    with open(RESULTS / "multiitem_recall_decoding_000469.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)

    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)
    stats["multiitem_recall_decoding_000469"] = out
    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(_json_safe(stats), f, indent=2, allow_nan=False)
    print("\nSaved results/multiitem_recall_decoding_000469.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
