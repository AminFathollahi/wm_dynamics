#!/usr/bin/env python3
"""Wolff et al. 2017 (Nature Neuroscience) human EEG impulse-perturbation dataset —
a direct test of activity-silent versus persistently-active working memory.

Cross-temporal generalization above chance throughout a delay period rules out
fully silent storage, but cannot distinguish a weakly-persistent code from a
genuinely silent, dynamically-read-out one. Wolff et al.'s impulse paradigm
provides a more direct test: a task-irrelevant visual perturbation is
introduced during the retention delay, and the memorandum's decodability from
the perturbation-evoked response (post-perturbation) is compared against its
decodability from the ongoing, unperturbed delay-period signal immediately
preceding the perturbation (pre-perturbation).

Experiment 1: two oriented gratings are shown (left and right), a retro-cue
indicates which orientation must be remembered, and a single task-irrelevant
impulse is flashed partway through the subsequent delay. Per-trial fields
(Results/Results_header): angle_left, angle_right (radians), cue (1=left,
2=right), probe_rotation (degrees), accuracy. Three EEG epochs are provided
per trial: EEG_mem_items (encoding-locked), EEG_cue (retro-cue-locked,
spanning the cue-to-impulse interval), and EEG_impulse (impulse-locked).

Experiment 2: two items are encoded per trial across two sessions, and each
item is probed in turn (early- versus late-tested, per the testing-order
condition), with one impulse preceding each probe (EEG_impulse1,
EEG_impulse2). No cue-locked epoch is recorded, so experiment 2 contributes
only a post-perturbation replication (for two distinct memoranda within the
same trial) rather than a pre- versus post-perturbation comparison.

Because the remembered orientation is a continuum, it is discretized into
six 30-degree bins spanning the fixed stimulus range [-pi/2, pi/2) and
decoded with the same multiclass, nested-cross-validation, label-permutation
pipeline (src/geometry.time_resolved_content_decoding) used for item-identity
decoding elsewhere in this project. Each epoch is reduced to a single
window-averaged feature vector (channels only) before decoding, rather than a
fine time-resolved sweep, for computational tractability; window boundaries
are chosen relative to each epoch's own onset (see WINDOW constants below)
and are necessarily approximate, since the exact cue-to-impulse and
impulse-to-probe stimulus onset asynchronies are not recorded in the
per-trial fields provided with this release.

Outputs: results/wolff_impulse_ctg.json
Updates: results/all_statistics.json — "wolff_impulse_ctg" key

Run:
    conda run -n wm_dynamics python scripts/run_wolff_impulse_pipeline.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.io as sio

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import time_resolved_content_decoding
from statistics import stable_seed, stouffer_combine, paired_sign_flip_test
from io_utils import locked_json_update

DATA_DIR = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/osfstorage/Wolff/data"
)
RESULTS = ROOT / "results"
MIN_SUBJECT_FILES = 20

N_ORIENTATION_BINS = 6
ORIENTATION_RANGE = (-np.pi / 2, np.pi / 2)
ENCODING_WINDOW_S = (0.1, 0.3)
PRE_PERTURBATION_WINDOW_S = (0.8, 1.098)
POST_PERTURBATION_WINDOW_S = (0.1, 0.3)
N_COMPONENTS = 10
N_SPLITS = 5
N_PERM = 500


def _load_mat(path: Path, n_retries: int = 3):
    """Load a .mat file, retrying on transient external-drive read failures.

    A single OSError ("could not read bytes") was observed and did not
    reproduce on an immediate re-read of the same file, indicating a
    transient I/O flake on the external drive rather than file corruption.
    """
    last_err = None
    for _ in range(n_retries):
        try:
            return sio.loadmat(str(path), struct_as_record=False, squeeze_me=True)
        except OSError as e:
            last_err = e
    raise last_err


def bin_orientation(angle: np.ndarray) -> np.ndarray:
    lo, hi = ORIENTATION_RANGE
    frac = np.clip((angle - lo) / (hi - lo), 0.0, 1.0 - 1e-9)
    return np.floor(frac * N_ORIENTATION_BINS).astype(int)


def window_features(epoch, window_s: tuple[float, float], valid: np.ndarray) -> np.ndarray:
    """(n_valid, n_channels, 1) window-averaged feature array for one epoch."""
    time = np.asarray(epoch.time)
    lo, hi = window_s
    t_mask = (time >= lo) & (time <= hi)
    trial = np.asarray(epoch.trial)[valid]
    feat = trial[:, :, t_mask].mean(axis=2)
    return feat[:, :, None]


def valid_trial_mask(epoch, n_trials: int) -> np.ndarray:
    mask = np.ones(n_trials, dtype=bool)
    bad = np.atleast_1d(epoch.bad_trials)
    if bad.size and np.isfinite(bad).all():
        mask[bad.astype(int) - 1] = False
    return mask


def decode_window(feat: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    res = time_resolved_content_decoding(
        feat, labels, t_idx=np.array([0]),
        n_components=N_COMPONENTS, n_splits=N_SPLITS, n_perm=N_PERM, rng=rng,
    )
    return {"effect": float(res["auc_per_t"][0] - 0.5), "p_value": float(res["p_per_t"][0])}


def analyze_subject_exp1(path: Path) -> dict:
    d = _load_mat(path)
    data = d["exp1_data"]
    results = data.Results
    n_trials = results.shape[0]
    remembered_angle = np.where(results[:, 2] == 1, results[:, 0], results[:, 1])
    labels_all = bin_orientation(remembered_angle)
    seed_key = path.stem

    out = {}
    epoch_specs = [
        ("encoding", data.EEG_mem_items, ENCODING_WINDOW_S),
        ("pre_perturbation", data.EEG_cue, PRE_PERTURBATION_WINDOW_S),
        ("post_perturbation", data.EEG_impulse, POST_PERTURBATION_WINDOW_S),
    ]
    for name, epoch, window in epoch_specs:
        valid = valid_trial_mask(epoch, n_trials)
        feat = window_features(epoch, window, valid)
        rng = np.random.default_rng(stable_seed(f"{seed_key}_{name}"))
        out[name] = decode_window(feat, labels_all[valid], rng)
    return out


def analyze_subject_exp2(path: Path) -> dict:
    d = _load_mat(path)
    data = d["exp2_data"]
    seed_key = path.stem

    out = {}
    epoch_specs = [
        ("encoding", ["EEG_mem_items_sess1", "EEG_mem_items_sess2"], ENCODING_WINDOW_S, 0),
        ("post_perturbation_early", ["EEG_impulse1_sess1", "EEG_impulse1_sess2"], POST_PERTURBATION_WINDOW_S, 0),
        ("post_perturbation_late", ["EEG_impulse2_sess1", "EEG_impulse2_sess2"], POST_PERTURBATION_WINDOW_S, 1),
    ]
    for name, fields, window, angle_col in epoch_specs:
        feats, labs = [], []
        for sess_idx, field in enumerate(fields, start=1):
            epoch = getattr(data, field)
            results = getattr(data, f"Results_sess{sess_idx}")
            n_trials = results.shape[0]
            angle = results[:, angle_col]
            valid = valid_trial_mask(epoch, n_trials)
            feats.append(window_features(epoch, window, valid))
            labs.append(bin_orientation(angle[valid]))
        feat = np.concatenate(feats, axis=0)
        labels = np.concatenate(labs, axis=0)
        rng = np.random.default_rng(stable_seed(f"{seed_key}_{name}"))
        out[name] = decode_window(feat, labels, rng)
    return out


def pool(per_subject: list[dict], key: str) -> dict:
    effects = np.array([s[key]["effect"] for s in per_subject])
    p_values = np.array([s[key]["p_value"] for s in per_subject])
    combined = stouffer_combine(p_values)
    return {
        "n_subjects": int(len(per_subject)),
        "mean_effect": float(effects.mean()),
        "z_combined": combined["z_combined"],
        "p_combined": combined["p_combined"],
    }


def main():
    exp1_files = sorted(DATA_DIR.glob("Dynamic_hidden_states_exp1_*.mat"))
    exp2_files = sorted(DATA_DIR.glob("Dynamic_hidden_states_exp2_*.mat"))
    if len(exp1_files) + len(exp2_files) < MIN_SUBJECT_FILES:
        print(f"SKIP - Wolff et al. 2017 data incomplete at {DATA_DIR} "
              f"({len(exp1_files)} exp1 + {len(exp2_files)} exp2 files found).")
        return

    print(f"Experiment 1: {len(exp1_files)} subjects")
    exp1_results = []
    for fp in exp1_files:
        try:
            r = analyze_subject_exp1(fp)
        except OSError as e:
            print(f"  {fp.stem}: SKIPPED after retries ({e})")
            continue
        exp1_results.append(r)
        print(f"  {fp.stem}: encoding={r['encoding']['effect']:.4f} (p={r['encoding']['p_value']:.4f}), "
              f"pre={r['pre_perturbation']['effect']:.4f} (p={r['pre_perturbation']['p_value']:.4f}), "
              f"post={r['post_perturbation']['effect']:.4f} (p={r['post_perturbation']['p_value']:.4f})")

    print(f"\nExperiment 2: {len(exp2_files)} subjects")
    exp2_results = []
    for fp in exp2_files:
        try:
            r = analyze_subject_exp2(fp)
        except OSError as e:
            print(f"  {fp.stem}: SKIPPED after retries ({e})")
            continue
        exp2_results.append(r)
        print(f"  {fp.stem}: encoding={r['encoding']['effect']:.4f} (p={r['encoding']['p_value']:.4f}), "
              f"post_early={r['post_perturbation_early']['effect']:.4f} (p={r['post_perturbation_early']['p_value']:.4f}), "
              f"post_late={r['post_perturbation_late']['effect']:.4f} (p={r['post_perturbation_late']['p_value']:.4f})")

    exp1_pooled = {
        "encoding": pool(exp1_results, "encoding"),
        "pre_perturbation": pool(exp1_results, "pre_perturbation"),
        "post_perturbation": pool(exp1_results, "post_perturbation"),
    }
    exp2_pooled = {
        "encoding": pool(exp2_results, "encoding"),
        "post_perturbation_early": pool(exp2_results, "post_perturbation_early"),
        "post_perturbation_late": pool(exp2_results, "post_perturbation_late"),
    }

    rng = np.random.default_rng(stable_seed("wolff_exp1_collapse_recover"))
    pre_effects = np.array([s["pre_perturbation"]["effect"] for s in exp1_results])
    post_effects = np.array([s["post_perturbation"]["effect"] for s in exp1_results])
    collapse_recover = paired_sign_flip_test(post_effects, pre_effects, alternative="greater", rng=rng)

    print("\nExperiment 1 pooled (Stouffer, N={}):".format(len(exp1_results)))
    for name, r in exp1_pooled.items():
        print(f"  {name}: mean_effect={r['mean_effect']:.4f}, z={r['z_combined']:.3f}, p={r['p_combined']:.4g}")
    print(f"  post > pre (paired sign-flip): mean_diff={collapse_recover['mean_diff']:.4f}, "
          f"p={collapse_recover['p_value']:.4g}")

    print("\nExperiment 2 pooled (Stouffer, N={}):".format(len(exp2_results)))
    for name, r in exp2_pooled.items():
        print(f"  {name}: mean_effect={r['mean_effect']:.4f}, z={r['z_combined']:.3f}, p={r['p_combined']:.4g}")

    out = {
        "n_orientation_bins": N_ORIENTATION_BINS,
        "windows_s": {
            "encoding": ENCODING_WINDOW_S,
            "pre_perturbation": PRE_PERTURBATION_WINDOW_S,
            "post_perturbation": POST_PERTURBATION_WINDOW_S,
        },
        "experiment1": {
            "per_subject": exp1_results,
            "pooled": exp1_pooled,
            "post_greater_than_pre": {
                "mean_diff": collapse_recover["mean_diff"],
                "p_value": collapse_recover["p_value"],
                "ci_lower": collapse_recover["ci_lower"],
                "ci_upper": collapse_recover["ci_upper"],
            },
        },
        "experiment2": {
            "per_subject": exp2_results,
            "pooled": exp2_pooled,
        },
    }
    import json
    with open(RESULTS / "wolff_impulse_ctg.json", "w") as f:
        json.dump(out, f, indent=2)
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["wolff_impulse_ctg"] = out
    print("\nSaved results/wolff_impulse_ctg.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
