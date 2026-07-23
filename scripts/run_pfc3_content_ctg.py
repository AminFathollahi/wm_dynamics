#!/usr/bin/env python3
"""CRCNS pfc-3 (Constantinidis lab) macaque PFC content-CTG — cross-species replication.

Single-neuron spike trains from macaque dlPFC during a spatial delayed-match-
-to-sample task (Meyer/Qi/Stanford/Constantinidis 2011; Kobak et al. 2016).
9 possible cue locations, held across a tightly fixed ~2 s cue-to-sample delay
(Cue_onT -> Sample_onT). Unlike Miller/Boran (load/context only) or Rutishauser
(item identity but only 5 trials/class after CV), this dataset gives a genuine
item-identity-during-delay test with a real memorandum (spatial location) in a
species/lab lineage independent of the two MTL datasets, providing a genuinely
independent replication that does not share the Boran/Rutishauser lineage's
task and methodology overlap.

Neurons were NOT recorded simultaneously, so a pseudo-population is built:
each pseudo-trial independently resamples one real trial per neuron per class
(standard practice for pseudo-simultaneous population decoding, as used in
the original dPCA paper this dataset was published with).

Outputs: results/pfc3_content_ctg.npz, updates results/all_statistics.json
  under "pfc3_content_ctg".

Run:
    conda run -n wm_dynamics python scripts/run_pfc3_content_ctg.py
"""
import sys, json, glob, random, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import scipy.io as sio
from geometry import ctg_content_permutation_null, temporal_stability_tau

DATA_DIR = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/PFC-3/extracted/data"
)
RESULTS = ROOT / "results"

WINDOW_S = 2.0            # delay duration: fixed ~2.03s Cue_onT -> Sample_onT
BIN_MS = 100
MIN_TRIALS_PER_CLASS = 8
N_NEURONS_TARGET = 80
M_PSEUDO = 15              # pseudo-trials per class
CTG_STEP = 2
N_SPLITS = 3
N_PERM = 10000   # headline content-CTG test (abstract/Discussion cite p<0.001)
SEED = 0


def load_neuron_spatial(fp: Path):
    """Return {class_idx (0-8): (n_trials, n_bins) rate array} or None if not
    a 9-class spatial-task file / has too few trials in any class."""
    try:
        d = sio.loadmat(str(fp), struct_as_record=False, squeeze_me=True)
        md = d["MatData"]
        if hasattr(md, "__len__"):
            md = md[0] if len(md) > 0 else md
        cls = md.__dict__["class"]
    except Exception:
        return None
    if not hasattr(cls, "__len__") or len(cls) != 9:
        return None

    n_bins = int(WINDOW_S * 1000 / BIN_MS)
    edges = np.linspace(0.0, WINDOW_S, n_bins + 1)
    out = {}
    for ci, c in enumerate(cls):
        ntr = c.ntr
        if not hasattr(ntr, "__len__"):
            ntr = [ntr]
        rates = []
        for t in ntr:
            try:
                cue, ts = t.Cue_onT, t.TS
            except Exception:
                continue
            if not np.isfinite(cue):
                continue
            aligned = np.atleast_1d(ts) - cue
            counts, _ = np.histogram(aligned, bins=edges)
            rates.append(counts / (BIN_MS / 1000.0))
        if len(rates) < MIN_TRIALS_PER_CLASS:
            return None
        out[ci] = np.stack(rates, axis=0)  # (n_trials, n_bins)
    return out


def main():
    files = sorted(glob.glob(str(DATA_DIR / "*.mat")))
    rng_py = random.Random(SEED)
    rng_py.shuffle(files)

    neurons = []
    for fp in files:
        rec = load_neuron_spatial(Path(fp))
        if rec is not None:
            neurons.append(rec)
        if len(neurons) >= N_NEURONS_TARGET:
            break
    n_neurons = len(neurons)
    print(f"Collected {n_neurons} spatial-task neurons "
          f"(>= {MIN_TRIALS_PER_CLASS} trials/class), scanned {files.index(fp) + 1} files")
    if n_neurons < 10:
        print("Too few qualifying neurons — aborting.")
        return

    n_bins = int(WINDOW_S * 1000 / BIN_MS)
    rng = np.random.default_rng(SEED)

    # Pseudo-population: (M_PSEUDO * 9, n_neurons, n_bins)
    X = np.zeros((M_PSEUDO * 9, n_neurons, n_bins), dtype=np.float32)
    y = np.zeros(M_PSEUDO * 9, dtype=int)
    row = 0
    for ci in range(9):
        for _ in range(M_PSEUDO):
            for ni, rec in enumerate(neurons):
                pool = rec[ci]
                pick = pool[rng.integers(0, pool.shape[0])]
                X[row, ni, :] = pick
            y[row] = ci
            row += 1

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sd

    t_idx = np.arange(0, n_bins, CTG_STEP)
    print(f"Pseudo-population: {X.shape[0]} pseudo-trials, {n_neurons} neurons, "
          f"{len(t_idx)} timepoints over [0, {WINDOW_S}]s delay")
    print(f"Running 9-way content-CTG (nested-CV, {N_PERM}-perm label-shuffle null)...")

    res = ctg_content_permutation_null(
        X, y, t_idx, n_components=min(20, n_neurons - 2),
        n_splits=N_SPLITS, n_perm=N_PERM, rng=rng,
    )
    tau_info = temporal_stability_tau(res["auc_mat"], min_diag_auc=0.55)

    print(f"Content-CTG (9-way spatial location, macaque PFC delay): "
          f"offdiag_effect={res['mean_offdiag_auc_minus_chance']:.4f}, "
          f"diag_effect={tau_info['diag_effect']:.4f}, "
          f"tau={tau_info['tau']:.4f} (interpretable={tau_info['interpretable']}), "
          f"p={res['p_value']:.4f}")

    np.savez_compressed(
        RESULTS / "pfc3_content_ctg.npz",
        auc_mat=res["auc_mat"],
        t_idx=t_idx,
        times=t_idx * BIN_MS / 1000.0,
        null=res["null"],
        n_neurons=n_neurons,
        n_classes=res["n_classes"],
        X=X, y=y,   # pseudo-population + labels, reused by the axis-rotation and dPCA analyses
    )

    stats_path = RESULTS / "all_statistics.json"
    with open(stats_path) as f:
        stats = json.load(f)
    stats["pfc3_content_ctg"] = {
        "dataset": "CRCNS pfc-3 (Constantinidis lab, macaque dlPFC, spatial DMS)",
        "n_neurons": n_neurons,
        "n_pseudo_trials": int(X.shape[0]),
        "n_classes": res["n_classes"],
        "window_s": WINDOW_S,
        "mean_offdiag_auc_minus_chance": res["mean_offdiag_auc_minus_chance"],
        "mean_diag_auc_minus_chance": res["mean_diag_auc_minus_chance"],
        "tau": tau_info["tau"],
        "tau_interpretable": tau_info["interpretable"],
        "p_value": res["p_value"],
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("Saved results/pfc3_content_ctg.npz, updated all_statistics.json")


if __name__ == "__main__":
    main()
