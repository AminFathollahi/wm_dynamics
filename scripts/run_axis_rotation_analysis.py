#!/usr/bin/env python3
"""Rotating (content) versus fixed (context) decoding-axis geometry.

The content/context temporal-stability dissociation (content tau substantially
below context tau) is a decoding-generalization pattern; this analysis asks
whether it has a geometric mechanism — does the classifier for context reuse
the same population direction throughout the delay (consistent with a
fixed-point attractor), while the classifier for content rotates onto
different directions over time (consistent with a ring-like or rotational
code)? geometry.coding_direction_stability addresses this directly via the
cosine similarity of the decoding weight vector w(t) across timepoints, which
is distinct from the question cross-temporal generalization answers ("can the
classifier generalize across time"): a decoder can generalize across time
either because a fixed component of the population pattern persists or
because the classifier effectively re-learns an equivalent decision boundary
at each timepoint. Coding-direction stability asks specifically about the
axis itself, and connects naturally to the DMD rotation frequency (the
imaginary part of the dominant complex eigenvalue is a rotation rate in the
same latent space).

Datasets:
  - content and context: DANDI 000469 (load axis, load 1 versus load 3, as
    context; item identity within load 1, as content)
  - content only: CRCNS pfc-3 (nine-way spatial location)
  - context only: Miller (0-back versus 2-back), Boran iEEG (set size 4 versus 8),
    Boran units (set size 4 versus 8, Round-7 STEP K1), DANDI 001187 and DANDI
    000673 (load 1 versus load 3, Round-7 STEP K1 — no repeated items in either,
    so context-axis only, matching the DATASET_ANALYSIS_MATRIX.md exclusion that
    keeps content-axis-rotation DANDI-000469-only, per the Fig-7 dissociation).

Outputs: results/axis_rotation_{dataset}.json
Updates: results/all_statistics.json — "axis_rotation_{dataset}" keys

Run (after run_000469_pipeline.py, run_pfc3_content_ctg.py, Miller/Boran geometry,
run_000574_units_pipeline.py, run_001187_pipeline.py, run_000673_pipeline.py):
    conda run -n wm_dynamics python scripts/run_axis_rotation_analysis.py
"""
import sys, json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import coding_direction_stability
from dynamics import ensemble_dmd
from spike_pipeline import fit_pca_psth
from statistics import spearman_permutation_test

RESULTS = ROOT / "results"
STEP_000469 = 3
STEP_PFC3 = 2
STEP_MILLER = 40
STEP_BORAN = 280
DMD_RANK = 7   # Round-7 STEP A: CV-selected primary rank (results/dmd_rank_selection.json)


def axis_rotation_index(cos_sim: np.ndarray) -> float:
    """1 - mean|cos| off-diagonal: 0 = perfectly fixed axis, ->1 = fully rotating."""
    n = cos_sim.shape[0]
    off = cos_sim[~np.eye(n, dtype=bool)]
    return float(1.0 - np.nanmean(off))


def rotation_frequency_hz(Z_trials: np.ndarray, dt: float, r: int = DMD_RANK) -> float:
    """Imaginary part (rad/s -> Hz) of the dominant (largest-magnitude) complex
    eigenvalue of the ensemble DMD operator fit to the same trials — the DMD
    analogue of "how fast does the population state rotate."""
    ens = ensemble_dmd(Z_trials, r=r, dt=dt, n_splits=3, n_null=10, rng=np.random.default_rng(0))
    lam = ens["eigenvalues"]
    dominant = lam[np.argmax(np.abs(lam))]
    omega = np.log(dominant + 1e-300) / dt   # continuous-time eigenvalue
    return float(np.abs(omega.imag) / (2 * np.pi))


def run_000469() -> dict:
    per_subject = {}
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]

        ctx_mask = (loads == 1) | (loads == 3)
        if ctx_mask.sum() < 10:
            continue
        ctx_labels = (loads[ctx_mask] == 3).astype(int)
        ctx_cos, ctx_t = coding_direction_stability(Z[ctx_mask], ctx_labels, step=STEP_000469)
        ctx_ari = axis_rotation_index(ctx_cos)

        load1_mask = loads == 1
        content_labels = pic_id[load1_mask]
        row = {"context_axis_rotation_index": ctx_ari, "n_context_trials": int(ctx_mask.sum())}
        if load1_mask.sum() >= 15 and len(np.unique(content_labels)) >= 2:
            content_cos, content_t = coding_direction_stability(
                Z[load1_mask], content_labels, step=STEP_000469
            )
            content_ari = axis_rotation_index(content_cos)
            dt = 0.1  # 100 ms bins (run_000469_pipeline.py BIN_MS)
            rot_freq = rotation_frequency_hz(Z[load1_mask], dt)
            row.update({
                "content_axis_rotation_index": content_ari,
                "content_dmd_rotation_freq_hz": rot_freq,
                "n_content_trials": int(load1_mask.sum()),
            })
        per_subject[subj] = row
    return per_subject


def run_pfc3() -> dict:
    d = np.load(RESULTS / "pfc3_content_ctg.npz", allow_pickle=True)
    X, y = d["X"], d["y"]   # (N, n_neurons, T)
    Z, _, _ = fit_pca_psth(X, n_comp=8)
    cos_sim, t_idx = coding_direction_stability(Z, y, step=STEP_PFC3)
    ari = axis_rotation_index(cos_sim)
    dt = 0.1  # 100 ms bins (run_pfc3_content_ctg.py BIN_MS)
    rot_freq = rotation_frequency_hz(Z, dt)
    return {"content_axis_rotation_index": ari, "content_dmd_rotation_freq_hz": rot_freq,
            "n_trials": int(X.shape[0])}


def run_miller() -> dict:
    per_subject = {}
    for subj in ["al", "ca", "cc", "ug"]:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id = d["Z"], d["task_id"]
        mask = (task_id == 0) | (task_id == 2)
        labels = (task_id[mask] == 2).astype(int)
        cos_sim, t_idx = coding_direction_stability(Z[mask], labels, step=STEP_MILLER)
        per_subject[subj] = {"context_axis_rotation_index": axis_rotation_index(cos_sim)}
    return per_subject


def run_boran() -> dict:
    per_subject = {}
    for path in sorted(RESULTS.glob("boran_geometry_sub-*.npz")):
        subj = path.stem.replace("boran_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, set_sizes = d["Z"], d["set_sizes"]
        mask = (set_sizes == 4) | (set_sizes == 8)
        if mask.sum() < 10:
            continue
        labels = (set_sizes[mask] == 8).astype(int)
        cos_sim, t_idx = coding_direction_stability(Z[mask], labels, step=STEP_BORAN)
        per_subject[subj] = {"context_axis_rotation_index": axis_rotation_index(cos_sim)}
    return per_subject


# ── Round-7 STEP K1: fillable gaps (context-axis only; see module docstring) ──
STEP_BINNED = 3   # same stride convention as STEP_000469 — these are all
                  # 100ms-bin PSTH latents (T~23-30), not raw-sample iEEG.
BIN_DT_S = 0.1


def run_boran_units() -> dict:
    """Boran units (DANDI 000574): set-size 4 vs 8, same context contrast as
    Boran iEEG, from the SAME subjects/trials — the within-subject spiking-vs-
    LFP dynamics comparison payoff (K3) needs this run at the matching contrast."""
    per_subject = {}
    for path in sorted(RESULTS.glob("dandi000574_units_geometry_sub-*.npz")):
        key = path.stem.replace("dandi000574_units_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, set_size = d["Z"], d["set_size"]
        mask = (set_size == 4) | (set_size == 8)
        if mask.sum() < 10:
            continue
        labels = (set_size[mask] == 8).astype(int)
        cos_sim, t_idx = coding_direction_stability(Z[mask], labels, step=STEP_BINNED)
        per_subject[key] = {"context_axis_rotation_index": axis_rotation_index(cos_sim),
                            "n_context_trials": int(mask.sum())}
    return per_subject


def _run_load1v3_context(glob_pattern: str, key_prefix: str) -> dict:
    """Shared K1 loader for DANDI 001187 / 000673: load-1-vs-load-3 context
    axis-rotation, same contrast run_000469's context axis uses. No repeated
    items in either cohort -> context only (see module docstring)."""
    per_session = {}
    for path in sorted(RESULTS.glob(glob_pattern)):
        key = path.stem.replace(key_prefix, "")
        d = np.load(path, allow_pickle=True)
        Z, loads = d["Z"], d["loads"]
        mask = (loads == 1) | (loads == 3)
        if mask.sum() < 10:
            continue
        labels = (loads[mask] == 3).astype(int)
        cos_sim, t_idx = coding_direction_stability(Z[mask], labels, step=STEP_BINNED)
        per_session[key] = {"context_axis_rotation_index": axis_rotation_index(cos_sim),
                            "n_context_trials": int(mask.sum())}
    return per_session


def run_dandi001187() -> dict:
    return _run_load1v3_context("dandi001187_geometry_sub-*.npz", "dandi001187_geometry_")


def run_dandi000673() -> dict:
    return _run_load1v3_context("dandi000673_geometry_sub-*.npz", "dandi000673_geometry_")


def main():
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)

    print("DANDI 000469 (content + context)...")
    d469 = run_000469()
    with open(RESULTS / "axis_rotation_dandi000469.json", "w") as f:
        json.dump(d469, f, indent=2)
    stats["axis_rotation_dandi000469"] = d469

    print("pfc-3 (content)...")
    pfc3 = run_pfc3()
    with open(RESULTS / "axis_rotation_pfc3.json", "w") as f:
        json.dump(pfc3, f, indent=2)
    stats["axis_rotation_pfc3"] = pfc3

    print("Miller (context)...")
    miller = run_miller()
    with open(RESULTS / "axis_rotation_miller.json", "w") as f:
        json.dump(miller, f, indent=2)
    stats["axis_rotation_miller"] = miller

    print("Boran iEEG (context)...")
    boran = run_boran()
    with open(RESULTS / "axis_rotation_boran.json", "w") as f:
        json.dump(boran, f, indent=2)
    stats["axis_rotation_boran"] = boran

    # Round-7 STEP K1: fillable gaps (context-axis only; see module docstring)
    print("Boran units (context, STEP K1)...")
    boran_units = run_boran_units()
    with open(RESULTS / "axis_rotation_boran_units.json", "w") as f:
        json.dump(boran_units, f, indent=2)
    stats["axis_rotation_boran_units"] = boran_units

    print("DANDI 001187 (context, STEP K1)...")
    d001187 = run_dandi001187()
    with open(RESULTS / "axis_rotation_dandi001187.json", "w") as f:
        json.dump(d001187, f, indent=2)
    stats["axis_rotation_dandi001187"] = d001187

    print("DANDI 000673 (context, STEP K1)...")
    d000673 = run_dandi000673()
    with open(RESULTS / "axis_rotation_dandi000673.json", "w") as f:
        json.dump(d000673, f, indent=2)
    stats["axis_rotation_dandi000673"] = d000673

    # Predict content > context axis-rotation index
    content_aris = [pfc3["content_axis_rotation_index"]] + [
        v["content_axis_rotation_index"] for v in d469.values() if "content_axis_rotation_index" in v
    ]
    context_aris = ([v["context_axis_rotation_index"] for v in d469.values()]
                    + [v["context_axis_rotation_index"] for v in miller.values()]
                    + [v["context_axis_rotation_index"] for v in boran.values()])
    print(f"\nContent axis-rotation index: mean={np.mean(content_aris):.3f} (N={len(content_aris)})")
    print(f"Context axis-rotation index: mean={np.mean(context_aris):.3f} (N={len(context_aris)})")

    # Primary within-subject test: DANDI 000469 has both axes from the same
    # subjects, so this is the cleanest paired comparison, following the same
    # design as the within-subject content-context temporal-stability comparison.
    from statistics import paired_sign_flip_test
    paired_subjs = {s: v for s, v in d469.items() if "content_axis_rotation_index" in v}
    within_subject_test = None
    if len(paired_subjs) >= 4:
        ctx = np.array([v["context_axis_rotation_index"] for v in paired_subjs.values()])
        cont = np.array([v["content_axis_rotation_index"] for v in paired_subjs.values()])
        res = paired_sign_flip_test(cont, ctx, n_perm=10000, alternative="greater",
                                    rng=np.random.default_rng(1))
        within_subject_test = {k: v for k, v in res.items() if k != "null"}
        within_subject_test["n_subjects"] = len(paired_subjs)
        print(f"\nWithin-subject (DANDI 000469, N={len(paired_subjs)}): "
              f"mean(content-context ARI)={res['mean_diff']:.4f} "
              f"[{res['ci_lower']:.4f}, {res['ci_upper']:.4f}], p={res['p_value']:.4f}")
    stats["axis_rotation_within_subject_dandi000469"] = within_subject_test

    # Payoff correlation: content-axis rotation rate vs DMD rotation frequency,
    # within DANDI 000469 subjects (the only dataset with both from the same trials)
    pairs = [(v["content_axis_rotation_index"], v["content_dmd_rotation_freq_hz"])
             for v in d469.values() if "content_dmd_rotation_freq_hz" in v]
    corr_result = None
    if len(pairs) >= 4:
        x = np.array([p[0] for p in pairs]); y = np.array([p[1] for p in pairs])
        corr_result = spearman_permutation_test(x, y, n_perm=10000, rng=np.random.default_rng(0))
        corr_result = {k: v for k, v in corr_result.items() if k != "null"}
        print(f"Content-axis rotation vs DMD rotation frequency (N={len(pairs)}): "
              f"rho={corr_result['rho']:.3f}, p={corr_result['p_value']:.4f}")
    stats["axis_rotation_vs_dmd_frequency_dandi000469"] = corr_result

    # Round-7 STEP K3: Boran units vs Boran iEEG -- spiking-vs-LFP within-subject
    # context-axis-rotation comparison (same subjects/trials, same set4v8 contrast).
    # Boran units is keyed per-session (sub-XX_ses-YY); average sessions to one
    # value per subject before pairing against Boran iEEG's per-subject values.
    units_by_subj: dict[str, list] = {}
    for key, v in boran_units.items():
        subj = key.split("_ses-")[0]
        units_by_subj.setdefault(subj, []).append(v["context_axis_rotation_index"])
    units_subj_mean = {s: float(np.mean(vs)) for s, vs in units_by_subj.items()}
    shared_subjs = sorted(set(units_subj_mean) & set(boran))
    spiking_vs_lfp = None
    if len(shared_subjs) >= 4:
        units_vals = np.array([units_subj_mean[s] for s in shared_subjs])
        ieeg_vals = np.array([boran[s]["context_axis_rotation_index"] for s in shared_subjs])
        res_k3 = paired_sign_flip_test(units_vals, ieeg_vals, n_perm=10000, alternative="two-sided",
                                       rng=np.random.default_rng(2))
        spiking_vs_lfp = {k: v for k, v in res_k3.items() if k != "null"}
        spiking_vs_lfp["n_subjects"] = len(shared_subjs)
        spiking_vs_lfp["subjects"] = shared_subjs
        spiking_vs_lfp["units_mean"] = float(units_vals.mean())
        spiking_vs_lfp["ieeg_mean"] = float(ieeg_vals.mean())
        print(f"\nSTEP K3 spiking-vs-LFP (Boran units vs Boran iEEG, context ARI, "
              f"N={len(shared_subjs)} shared subjects): units_mean={units_vals.mean():.4f} "
              f"ieeg_mean={ieeg_vals.mean():.4f} diff={res_k3['mean_diff']:+.4f} "
              f"[{res_k3['ci_lower']:.4f}, {res_k3['ci_upper']:.4f}] p={res_k3['p_value']:.4f} "
              f"({'AGREE (no reliable diff)' if res_k3['p_value'] >= 0.05 else 'DIVERGE'})")
    else:
        print(f"\nSTEP K3 spiking-vs-LFP: only {len(shared_subjs)} shared subjects "
              f"(<4) -- underpowered, STOP-and-report, not computed.")
    stats["axis_rotation_spiking_vs_lfp_boran"] = spiking_vs_lfp

    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved axis_rotation_*.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
