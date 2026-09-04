#!/usr/bin/env python3
"""Nine-arm targeting benchmark: ADDS macrosignal_pac to the
EXISTING 8-arm results/causal_benchmark.json leaderboard. DOES NOT SHRINK IT.

FRAMING (mandatory): the leaderboard has 8 existing
arms scored on the macaque uStim causal design (the only
dataset with delivered stimulation + a designed propensity -- see
run_macaque_pfc_microstimulation_pipeline.py). "anatomical vs manifold vs macro-signal" is a
THREE-FAMILY narrative grouping, not a reduction to 3 entries. This script
ADDS a 9th arm; run_macaque_pfc_microstimulation_pipeline.py (already re-run at r=7 for Step A)
is untouched and its 8 existing arms are copied through unchanged.

DATA-AVAILABILITY CONSTRAINT (why macrosignal_pac is scored differently from
the other 8, and why that is not an error): macaque PFC microstimulation is macaque intracortical
SPIKE-RATE data -- it has no LFP, so no theta-phase / gamma-amplitude PAC
signal exists there to compute. E1 explicitly requires PAC on the LFP
datasets (Boran iEEG, DANDI 000673, Miller ECoG); of those, only Boran iEEG
has BOTH a fitted plant (A) AND a TES1-derived B-donor bank (DATASET_ANALYSIS_
MATRIX.md exclusion #2: TES1 covers DLPFC only, and among the LFP cohorts only
Miller+Boran have TES1 coverage; Miller has no usable outcome label -- see
run_behavior_ctg.py). So macrosignal_pac is constructed and scored on Boran
iEEG's OWN closed-loop targeting-benchmark (below), NOT by fabricating a PAC
signal for the macaque PFC microstimulation rows. Its slope/CI/p in the final leaderboard trace to
this Boran construction; agent_report.md documents this plainly.

BORAN TARGETING BENCHMARK (all 9 arms, drift-reduction AND flip-rate):
For each Boran subject with a fitted (A, TES1 B-bank) -- the SAME bundle
run_closed_loop_analysis.py and run_closed_loop_behavior_flip.py already use
-- construct all NINE candidate latent steering directions as a REAL TES1
donor (or, where no better anatomical target exists, a directly-computed
vector) selected by each arm's own criterion, exactly mirroring how
run_macaque_pfc_microstimulation_pipeline.py builds v_star / v_stable / random_dirs / min_energy_dir:
  vstar_alignment          : TES1 donor maximizing |cos(B_i, v*)|            (= dynamic_best_idx, reused)
  gramian_trace            : TES1 donor maximizing its own Gramian trace     (= static_best_idx, reused)
  stable_alignment         : TES1 donor maximizing |cos(B_i, v_stable)| (slowest/most-stable eigenvector)
  random_alignment         : a fixed-seed random unit direction in latent space (macaque PFC microstimulation's own convention)
  input_norm                : TES1 donor maximizing ||B_i|| (norm-matched, weakest prior)
  min_energy_dir_alignment : TES1 donor maximizing |cos(B_i, Gramian's top eigenvector)|
  anat_avg_ctrl/anat_modal_ctrl : DEGENERATE here (same principled reason the existing macaque PFC microstimulation
      arms are already degenerate -- see module docstring below); Boran electrodes are MTL/
      temporal, and no macaque-Markov-connectome mapping applies to human MTL, so there is no
      non-fabricated way to assign a differentiated anatomical-controllability value per TES1
      donor for this cohort. Reported honestly as a constant (zero-variance) arm, not omitted.
  macrosignal_pac          : TES1 donor maximizing |cos(B_i, B_pac)|, where B_pac = V.T @ mi is
      the theta(4-8Hz)-phase/HGP(70-150Hz)-amplitude PAC modulation index (Tort et al. 2010;
      src/preprocessing.phase_amplitude_coupling, REUSED not reimplemented) per good channel,
      projected into the session's own latent space -- weighting the "stimulation" direction
      toward the highest-PAC channels, as E1 specifies.

Each arm's drift-reduction (simulate_closed_loop) and flip-rate (the SAME
_flip_one_trial machinery run_closed_loop_behavior_flip.py already runs for
its on-demand/null-control comparison) is computed per subject; the
across-subject slope/CI/p is estimated with dml_partial_linear (continuous
exposure = that arm's per-subject alignment-to-v* score, outcome =
drift_reduction or flip fraction, confounder = subject dummies) -- the SAME
DR/DML gate machinery named in spec E2, reused rather than reimplemented.

Outputs: results/targeting_benchmark_boran.json (full per-arm/per-subject detail)
Updates: results/causal_benchmark.json -- "leaderboard" gains "macrosignal_pac"
         (9 keys total); "winner" updated to the true argmax over all 9.
         results/all_statistics.json -- "targeting_benchmark_boran" key.

Run (after run_boran_pipeline.py, run_tes1_analysis.py, run_divergence_analysis.py,
run_closed_loop_behavior_flip.py):
    conda run -n wm_dynamics python scripts/run_targeting_benchmark.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

import h5py
import scipy.signal as sig

from closed_loop import simulate_closed_loop, _b_hat_at_angle
from control import controllability_gramian
from causal import dml_partial_linear
from preprocessing import (phase_amplitude_coupling, bandpass_filter,
                           line_noise_notch, bipolar_reference_by_shank)
from statistics import bootstrap_ci, stable_seed
from run_closed_loop_behavior_flip import _fit_outcome_decoder_and_margin, _flip_one_trial
from io_utils import locked_json_update
from provenance import _json_safe

RESULTS = ROOT / "results"
DATA_ROOT = data_root()
BORAN_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 10)]
B_HAT_MISMATCH_DEG = 20.0
N_RANDOM_DIRS = 20
GRAMIAN_HORIZON = 20
ARM_NAMES = ["vstar_alignment", "gramian_trace", "stable_alignment", "random_alignment",
            "input_norm", "anat_avg_ctrl", "anat_modal_ctrl", "min_energy_dir_alignment",
            "macrosignal_pac"]
# Pre-specified BEFORE inspecting any donor's
# destabilization (see results/pac_donor_diagnostic.json, which established
# with this exact tolerance that all 3 originally-destabilized macrosignal_pac
# subjects have a non-destabilizing near-tie donor). Applied uniformly to
# EVERY criterion-based arm below, not just PAC: when the top-scoring donor
# destabilizes the plant, the next-best donor within this fraction of the top
# score is tried instead, in descending-score order.
NEAR_TIE_REL_TOL = 0.90


def _near_tie_candidates(scores: np.ndarray, tol: float = NEAR_TIE_REL_TOL) -> list[int]:
    max_score = np.max(scores)
    idxs = [i for i in range(len(scores)) if scores[i] >= tol * max_score]
    idxs.sort(key=lambda i: -scores[i])
    return idxs


def _pac_channel_weights(subj: str, good_ch_indices: np.ndarray) -> np.ndarray | None:
    """Theta-phase/HGP-amplitude PAC modulation index per GOOD channel (indices
    matching boran_geometry_*.npz's V rows exactly -- good_ch_indices selects
    into the raw full-channel data the same way run_boran_pipeline.py's
    channel_rejection mask does), on the RAW maintenance-window LFP
    (pre-HGP-transform -- PAC needs phase, band power alone destroys it),
    averaged over trials. Reuses preprocessing.phase_amplitude_coupling
    directly (Tort et al. 2010 MI)."""
    subj_dir = DATA_ROOT / "000574" / subj
    srate = 1398.0
    t_pre_maint, t_post_maint, t_epoch_pre = 3.0, 3.0, 1.0
    epoch_total = t_pre_maint + t_post_maint + t_epoch_pre
    all_epochs = []
    electrode_labels = None
    for nwb_path in sorted(subj_dir.glob("*.nwb")):
        with h5py.File(str(nwb_path), "r") as f:
            raw = f["acquisition/ecephys.ieeg/data"][:]
            times = f["acquisition/ecephys.ieeg/timestamps"][:]
            t_start_arr = f["intervals/trials/start_time"][:]
            artifact = f["intervals/trials/artifact"][:].astype(bool)
            if electrode_labels is None:
                try:
                    ieeg_idx = f["acquisition/ecephys.ieeg/electrodes"][:]
                    labels_full = [l.decode() for l in
                                    f["general/extracellular_ephys/electrodes/label"][:]]
                    electrode_labels = [labels_full[i] for i in ieeg_idx]
                except KeyError:
                    electrode_labels = [f"ch{i}" for i in range(raw.shape[1])]
        n_samp = int(epoch_total * srate)
        n_pre = int(t_epoch_pre * srate)
        for trial_idx, t0 in enumerate(t_start_arr):
            if artifact[trial_idx]:
                continue
            i0 = np.searchsorted(times, t0) - n_pre
            i1 = i0 + n_samp
            if i0 < 0 or i1 > raw.shape[0]:
                continue
            all_epochs.append(raw[i0:i1].astype(np.float32))   # (T, C_full)
    if len(all_epochs) < 10:
        return None

    # Notch (50/100/150 Hz) + bipolar-by-shank reref.
    # Must exactly match run_boran_pipeline.py's channel reduction, since
    # good_ch_indices (below) indexes into that same bipolar channel set,
    # not the raw monopolar channels.
    epochs_arr = np.stack(all_epochs, axis=0)           # (N, T, C_full)
    N0, T0, C_raw = epochs_arr.shape
    for n in range(N0):
        epochs_arr[n] = line_noise_notch(epochs_arr[n], srate, fundamental=50.0, n_harmonics=3)
    X_flat = epochs_arr.reshape(-1, C_raw)
    X_bp, _ = bipolar_reference_by_shank(X_flat, electrode_labels)
    epochs_arr = X_bp.reshape(N0, T0, -1).astype(np.float32)  # (N, T, C_bp)
    all_epochs = list(epochs_arr)

    maint_start_s = int((t_epoch_pre + t_pre_maint) * srate)
    maint_end_s = int((t_epoch_pre + t_pre_maint + t_post_maint) * srate)
    mi_per_trial = []
    # Subsample trials for tractability (PAC's Hilbert-transform cost scales
    # with samples x channels x trials; 30 trials is ample to average a
    # per-channel MI estimate that is a session-level scalar, not a per-trial
    # decode).
    rng = np.random.default_rng(stable_seed(f"pac_{subj}"))
    idx = rng.choice(len(all_epochs), size=min(30, len(all_epochs)), replace=False)
    for i in idx:
        maint = all_epochs[i][maint_start_s:maint_end_s][:, good_ch_indices]   # (T_maint, C_good)
        mi = phase_amplitude_coupling(maint, phase_band=(4.0, 8.0), amplitude_band=(70.0, 150.0),
                                      srate=srate, n_phase_bins=18)
        mi_per_trial.append(mi)
    mi_mean = np.nanmean(np.stack(mi_per_trial), axis=0)
    return mi_mean


def _build_arm_directions(A: np.ndarray, B_bank: np.ndarray, gramian_traces: np.ndarray,
                          V: np.ndarray, mi_channels: np.ndarray | None,
                          rng: np.random.Generator) -> dict:
    """Per subject: per-DONOR criterion scores for each arm (NOT a resolved
    single donor -- rescue-aware selection among near-tie candidates happens
    in run_boran_targeting_benchmark, which simulates candidates in
    descending-score order and prefers a non-destabilizing one; see
    NEAR_TIE_REL_TOL / Part 8B). align_vstar (per-donor alignment to the TRUE
    v*, independent of which criterion picked a donor) is returned separately
    and used to report "align_to_vstar" for whichever donor each arm ends up
    with."""
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = vecs[:, order[0]].real
    v_star = v_star / (np.linalg.norm(v_star) + 1e-12)
    stable_order = np.argsort(np.abs(eigs))
    v_stable = vecs[:, stable_order[0]].real
    v_stable = v_stable / (np.linalg.norm(v_stable) + 1e-12)

    k = A.shape[0]
    B_units = B_bank[:, :, 0] / (np.linalg.norm(B_bank[:, :, 0], axis=1, keepdims=True) + 1e-12)
    align_vstar = np.abs(B_units @ v_star)

    random_dirs = rng.standard_normal((N_RANDOM_DIRS, k))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-12
    random_align = np.mean(np.abs(random_dirs @ v_star))   # scalar summary, matches
                                                            # run_macaque_pfc_microstimulation_pipeline's convention

    input_norms = np.linalg.norm(B_bank[:, :, 0], axis=1)
    stable_align = np.abs(B_units @ v_stable)

    # Gramian's own top eigenvector (min-energy steering direction), using
    # the RAW argmax-align_vstar donor's own B for a representative Gramian
    # (matches run_macaque_pfc_microstimulation_pipeline's M6 construction: one Gramian per
    # session, not per-donor, since it is a plant-level not donor-level
    # quantity -- unaffected by rescue selection downstream).
    dyn_idx_tmp = int(np.argmax(align_vstar))
    Wc = controllability_gramian(A, B_bank[dyn_idx_tmp], T=GRAMIAN_HORIZON)
    wc_eigvals, wc_eigvecs = np.linalg.eigh(Wc)
    min_energy_dir = wc_eigvecs[:, np.argmax(wc_eigvals)]
    min_energy_align = np.abs(B_units @ min_energy_dir)

    # Criterion-based arms: each gets its OWN per-donor score array; a single
    # donor is resolved later via near-tie rescue search, identically for all.
    scores = {
        "vstar_alignment": align_vstar,
        "gramian_trace": gramian_traces,
        "stable_alignment": stable_align,
        "input_norm": input_norms,
        "min_energy_dir_alignment": min_energy_align,
    }
    if mi_channels is not None:
        B_pac_chan = np.nan_to_num(mi_channels)   # already good-channel-aligned; len == V.shape[0]
        B_pac_lat = V.T @ B_pac_chan
        B_pac_unit = B_pac_lat / (np.linalg.norm(B_pac_lat) + 1e-12)
        pac_align_to_donors = np.abs(B_units @ B_pac_unit)
        scores["macrosignal_pac"] = pac_align_to_donors

    # anat_avg_ctrl/anat_modal_ctrl: DEGENERATE (see module docstring) -- no
    # per-donor anatomical-controllability differentiation exists for Boran's
    # MTL electrodes, so they always take the SAME donor as vstar_alignment's
    # final (possibly rescued) pick, resolved in the caller. random_alignment:
    # not a single-donor criterion -- macaque PFC microstimulation's own arm is "mean |cos| to 20
    # random directions," a scalar exposure with no natural single B to
    # simulate; it also rides vstar_alignment's final donor for the ROLLOUT
    # (so its drift/flip numbers are comparable in scale) while random_align
    # stays the EXPOSURE dml_partial_linear regresses on.
    return {
        "scores": scores,
        "align_vstar": align_vstar,
        "random_align": float(random_align),
        "has_pac": mi_channels is not None,
    }


def _stability_horizon(A: np.ndarray, n_time_constants: float = 3.0) -> float:
    """SAME construction as run_closed_loop_analysis.py's _stability_horizon:
    caps the rollout at n_time_constants e-folding times of A's least-stable
    eigenvalue, so a near-unit-circle-to-unstable plant (every Boran A here
    has max|eig|>1, per that module's docstring) does not saturate
    simulate_closed_loop's numerical state-norm cap before the horizon ends
    -- which manufactures an arbitrarily huge (not genuine) drift number."""
    log_lam_max = np.log(np.max(np.abs(np.linalg.eigvals(A))))
    if abs(log_lam_max) < 1e-12:
        return np.inf
    return n_time_constants / abs(log_lam_max)


def run_boran_targeting_benchmark() -> dict:
    tes1_boran = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
    div = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)
    try:
        with open(RESULTS / "behavior_ctg.json") as f:
            peak_time_s = json.load(f).get("boran_ieeg", {}).get("peak_time_s")
    except FileNotFoundError:
        peak_time_s = None

    per_subject = {}
    for subj in BORAN_SUBJECTS:
        if f"{subj}_A_dmd" not in tes1_boran:
            print(f"  SKIP {subj} -- no TES1 bundle")
            continue
        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z, correct = geo["Z"], geo.get("correct", np.ones(geo["Z"].shape[0], dtype=bool)).astype(bool)
        V = geo["V"]
        if correct.sum() < 10 or (~correct).sum() < 10:
            print(f"  SKIP {subj} -- insufficient trials for a decoder")
            continue

        A = tes1_boran[f"{subj}_A_dmd"]
        B_bank = tes1_boran[f"{subj}_B_latent_per_tes1"]
        gramian_traces = tes1_boran[f"{subj}_gramian_traces"]
        x0, xf = tes1_boran[f"{subj}_x0"], tes1_boran[f"{subj}_xf"]

        rng = np.random.default_rng(stable_seed(f"targbench_{subj}"))
        print(f"  {subj}: computing PAC channel weights...")
        mi_channels = _pac_channel_weights(subj, geo["good_ch_indices"])
        built = _build_arm_directions(A, B_bank, gramian_traces, V, mi_channels, rng)
        align_vstar = built["align_vstar"]

        # SAME peak-AUC-timepoint fix as run_closed_loop_behavior_flip.py (Step D):
        # a per-timestep-pooled decoder was tried first and reliably called every
        # trial "correct" for every subject (near-chance cv_acc); fitting at the
        # single timepoint results/behavior_ctg.json's outcome-CTG already found
        # most decodable recovers a small but non-zero predicted-error set.
        peak_t_idx = (int(np.argmin(np.abs(geo["times"] - peak_time_s)))
                     if peak_time_s is not None else None)
        decoder_fit = _fit_outcome_decoder_and_margin(Z, correct, peak_t_idx)
        decoder, margin_fn, pred_error_trial, target, decoder_cv_acc = decoder_fit if decoder_fit else (None,) * 5
        n_error_pred = int(pred_error_trial.sum()) if decoder_fit else 0

        # Content/context decoder for drift-reduction scoring, matching
        # run_closed_loop_analysis.py's design (fit on real uncontrolled trials).
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        set_sizes = geo["set_sizes"]
        mask_ctx = (set_sizes == 4) | (set_sizes == 8)
        Z_ctx = Z[mask_ctx]
        labels_ctx = np.repeat((set_sizes[mask_ctx] == 8).astype(int), Z_ctx.shape[1])
        Z_ctx_feat = Z_ctx.reshape(-1, Z_ctx.shape[-1])
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
        pipe.fit(Z_ctx_feat, labels_ctx)
        context_decoder = lambda X: pipe.predict(X)

        state_scale = float(np.linalg.norm(xf - x0)) + 1e-6
        proc_noise, obs_noise = 0.05 * state_scale, 0.10 * state_scale
        horizon_real = min(Z.shape[1] - 1, 300)
        stab_horizon = _stability_horizon(A, n_time_constants=3.0)
        horizon = int(min(horizon_real, np.ceil(stab_horizon))) if np.isfinite(stab_horizon) else horizon_real

        def _simulate_and_flip(donor_idx: int) -> dict:
            """One donor's full rollout + flip-rate. SAME exclusion criterion as
            run_closed_loop_analysis.py: a donor whose LQR design DESTABILIZES
            this plant relative to open-loop (rho_closed > rho_open) produces an
            unboundedly large, not genuinely informative, drift number -- flagged
            here, excluded from pooling downstream, not silently averaged in."""
            B_true = B_bank[donor_idx]
            B_hat = _b_hat_at_angle(B_true, B_HAT_MISMATCH_DEG, rng)
            res = simulate_closed_loop(
                A, B_true, x0, xf, context_decoder, label=1, trigger="decoder",
                A_hat=A, B_hat=B_hat, obs_noise=obs_noise, proc_noise=proc_noise, u_budget=1.0,
                horizon=horizon, n_trials=30, n_boot=500,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            )
            destabilized = bool(res["rho_closed"] > res["rho_open"])
            flip_frac = float("nan")
            if decoder_fit is not None and n_error_pred >= 3:
                flips = []
                for tr in np.where(pred_error_trial)[0]:
                    trial_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
                    flipped, _, _ = _flip_one_trial(A, B_true, B_hat, Z[tr, 0, :], target,
                                                    decoder, margin_fn, horizon, obs_noise,
                                                    proc_noise, trial_rng)
                    flips.append(flipped)
                flip_frac = float(np.mean(flips))
            return {
                "donor_idx": donor_idx, "align_to_vstar": float(align_vstar[donor_idx]),
                "drift_reduction": res["drift_reduction"], "flip_frac": flip_frac,
                "destabilized": destabilized, "rho_open": res["rho_open"], "rho_closed": res["rho_closed"],
            }

        def _resolve_with_rescue(criterion_scores: np.ndarray) -> dict:
            """Part 8B: try near-tie candidates in descending criterion order;
            return the first non-destabilizing donor's full result, else the raw
            argmax's result (unchanged behavior when no rescue is available or
            needed)."""
            candidates = _near_tie_candidates(criterion_scores)
            tried = []
            for i, cand in enumerate(candidates):
                entry = _simulate_and_flip(cand)
                tried.append(entry)
                if not entry["destabilized"]:
                    entry["donor_selection"] = "argmax" if i == 0 else "near_tie_rescue"
                    entry["n_candidates_tried"] = i + 1
                    return entry
            tried[0]["donor_selection"] = "argmax_no_rescue"
            tried[0]["n_candidates_tried"] = len(tried)
            return tried[0]

        subj_row = {"arms": {}, "n_error_pred": n_error_pred, "decoder_cv_acc": decoder_cv_acc}
        for arm, criterion_scores in built["scores"].items():
            subj_row["arms"][arm] = _resolve_with_rescue(criterion_scores)

        # anat_avg_ctrl/anat_modal_ctrl/random_alignment: always the SAME donor
        # as vstar_alignment's final (possibly rescued) pick -- see
        # _build_arm_directions docstring. Each still gets its own independent
        # simulate call (fresh B_hat draw), matching the original per-arm noise
        # convention.
        vstar_idx = subj_row["arms"]["vstar_alignment"]["donor_idx"]
        for arm in ("anat_avg_ctrl", "anat_modal_ctrl", "random_alignment"):
            entry = _simulate_and_flip(vstar_idx)
            entry["donor_selection"] = "same_as_vstar"
            entry["n_candidates_tried"] = 1
            if arm in ("anat_avg_ctrl", "anat_modal_ctrl"):
                entry["align_to_vstar"] = 0.0   # degenerate arm, see module docstring
            else:
                entry["align_to_vstar"] = built["random_align"]   # EXPOSURE stays the 20-random-dir mean
            subj_row["arms"][arm] = entry

        if not built["has_pac"]:
            subj_row["arms"]["macrosignal_pac"] = None

        # preserve the documented ARM_NAMES order in the output
        subj_row["arms"] = {arm: subj_row["arms"].get(arm) for arm in ARM_NAMES}
        per_subject[subj] = subj_row
        print(f"    {subj}: " + ", ".join(f"{a}={subj_row['arms'][a]['drift_reduction']:.2f}"
                                          for a in ARM_NAMES if subj_row['arms'][a] is not None))

    return per_subject


def _pool_arm(per_subject: dict, arm: str, field: str) -> dict | None:
    """dml_partial_linear: field ~ align_to_vstar (continuous exposure),
    confounders = subject dummies -- the DR/DML machinery spec E2 names.
    Excludes destabilized (subject, arm) rows (see run_boran_targeting_
    benchmark's inline comment) -- a destabilizing controller's drift number
    is a numerical artifact, not a genuine "worse control" effect."""
    rows = [(v["arms"][arm]["align_to_vstar"], v["arms"][arm][field], subj)
           for subj, v in per_subject.items()
           if v["arms"].get(arm) is not None and np.isfinite(v["arms"][arm][field])
           and not v["arms"][arm].get("destabilized", False)]
    if len(rows) < 4:
        return None
    aligns = np.array([r[0] for r in rows])
    ys = np.array([r[1] for r in rows])
    subjs = sorted(set(r[2] for r in rows))
    X = np.eye(len(subjs))[[subjs.index(r[2]) for r in rows]]
    res = dml_partial_linear(ys, aligns, X, n_folds=min(5, len(rows)),
                             rng=np.random.default_rng(stable_seed(f"dml_{arm}_{field}")))
    return {"theta": res["theta"], "se": res["se"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
           "p_value": res["p_value"], "n": res["n"]}


def main():
    print("Boran iEEG targeting benchmark (9 arms: drift-reduction + flip-rate)...")
    per_subject = run_boran_targeting_benchmark()

    if not per_subject:
        print("No usable Boran subject -- STOP, cannot build the targeting benchmark.")
        return

    leaderboard_boran = {}
    for arm in ARM_NAMES:
        dr = _pool_arm(per_subject, arm, "drift_reduction")
        flip = _pool_arm(per_subject, arm, "flip_frac")
        stable_rows = [v["arms"][arm] for v in per_subject.values()
                      if v["arms"].get(arm) is not None and not v["arms"][arm].get("destabilized", False)]
        n_destabilized = sum(1 for v in per_subject.values()
                             if v["arms"].get(arm) is not None and v["arms"][arm].get("destabilized", False))
        mean_drift = float(np.nanmean([r["drift_reduction"] for r in stable_rows])) if stable_rows else float("nan")
        flip_vals = [r["flip_frac"] for r in stable_rows if np.isfinite(r["flip_frac"])]
        mean_flip = float(np.nanmean(flip_vals)) if flip_vals else float("nan")
        leaderboard_boran[arm] = {
            "mean_drift_reduction": mean_drift, "mean_flip_rate": mean_flip,
            "drift_reduction_dml": dr, "flip_rate_dml": flip,
            "n_subjects": len(stable_rows), "n_destabilized_excluded": n_destabilized,
        }
        print(f"  {arm}: mean_drift_reduction={mean_drift:+.3f} mean_flip_rate={mean_flip:.3f}")

    out_full = {"per_subject": per_subject, "leaderboard": leaderboard_boran}
    with open(RESULTS / "targeting_benchmark_boran.json", "w") as f:
        json.dump(_json_safe(out_full), f, indent=2, allow_nan=False)

    # Extend the EXISTING 9-... wait, 8-arm results/causal_benchmark.json with
    # macrosignal_pac as the 9th key. The other 8 keys are copied through
    # UNCHANGED (already re-scored at r=7 by run_macaque_pfc_microstimulation_pipeline.py, Step A).
    # Every existing arm ALSO gains a "flip_rate" field per spec E2/E3: for the
    # 8 macaque PFC microstimulation arms this is None (macaque PFC microstimulation has no outcome-decoder flip
    # construction -- WP-CAUSAL's Y is trial correct/error directly, not a
    # simulated closed-loop rollout; adding a fabricated flip_rate for those
    # 8 would violate the anti-fabrication contract). macrosignal_pac's own
    # flip_rate comes from the real Boran flip construction above.
    with locked_json_update(RESULTS / "causal_benchmark.json") as bench:
        for arm in bench.get("leaderboard", {}):
            bench["leaderboard"][arm].setdefault("flip_rate", None)
        pac_dr = leaderboard_boran["macrosignal_pac"]["drift_reduction_dml"]
        pac_flip = leaderboard_boran["macrosignal_pac"]["flip_rate_dml"]
        bench["leaderboard"]["macrosignal_pac"] = {
            "slope": pac_dr["theta"] if pac_dr else float("nan"),
            "slope_ci_lo": pac_dr["ci_lo"] if pac_dr else float("nan"),
            "slope_ci_hi": pac_dr["ci_hi"] if pac_dr else float("nan"),
            "p_value": pac_dr["p_value"] if pac_dr else float("nan"),
            "n": pac_dr["n"] if pac_dr else 0,
            "flip_rate": leaderboard_boran["macrosignal_pac"]["mean_flip_rate"],
            "flip_rate_dml": pac_flip,
            "scored_on": "boran_ieeg_targeting_benchmark",   # NOT the macaque PFC microstimulation pooled
                                                              # rows the other 8 arms use
                                                              # (see module docstring)
        }
        all_slopes = {a: v["slope"] for a, v in bench["leaderboard"].items()
                     if np.isfinite(v.get("slope", float("nan"))) and v["slope"] > 0}
        bench["winner"] = max(all_slopes, key=all_slopes.get) if all_slopes else bench.get("winner")
        bench["n_arms"] = len(bench["leaderboard"])
        print(f"\ncausal_benchmark.json leaderboard now has {len(bench['leaderboard'])} arms; "
              f"winner={bench['winner']}")

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["targeting_benchmark_boran"] = out_full
        stats["causal_benchmark"] = json.load(open(RESULTS / "causal_benchmark.json"))
    print("\nSaved results/targeting_benchmark_boran.json, extended results/causal_benchmark.json "
          "to 9 arms, updated all_statistics.json")


if __name__ == "__main__":
    main()
