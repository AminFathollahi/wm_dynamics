#!/usr/bin/env python3
"""Soldado-Magraner et al. 2025 J Neurosci — macaque dlPFC delay-period microstim.

CAUSAL KEYSTONE dataset for the Phase-1 GO/NO-GO gate: this is the only
dataset in the project where stimulation is delivered DURING the WM delay
with a designed (near-known) propensity, letting src/causal.py's doubly-
robust CATE-vs-modifier test run in its ideal regime.

Two monkeys, two incompatible MAT-file generations in the same data release:
  - Sa (1 session): MATLAB v7.3 (HDF5-backed) -- bipolar stim pairs, 192 ch
    (2 Utah arrays), ASCII-code-encoded char arrays, (time, channel) cell
    orientation after h5py's implicit transpose.
  - Wa (10 sessions): MATLAB v5/v7 (scipy-loadable) -- single-channel stim,
    96 ch (1 array), plain string arrays, (channel, time) cell orientation.
Both are parsed into one common per-trial representation before any
numerical work; the numerical work itself (PCA, DMD, v*, Gramian, CATE) is
100% shared with the rest of the project (src/geometry.py, src/dynamics.py,
src/control.py, src/causal.py) -- no new geometry/causal logic here.

Design (WP-CAUSAL-1/1b/2/3, Phase 1 of comments.txt Sec 6):
  1. Per session: crop every trial's spikerate to a common stim-onset-aligned
     window, split into CONTROL (no stim) vs STIM conditions.
  2. Fit PCA + DMD on the CONTROL, correct trials only (the plant is
     identified from unperturbed behaviour, never from stim/error trials).
  3. For each STIM condition, build a one-hot "input direction" B at the
     stimulated electrode(s), project into the control-fit latent space, and
     take |cos(B, v*)| as the pre-registered effect modifier.
  4. Pool every (stim-condition vs. that session's shared control pool)
     comparison across sessions into one long table and call the EXISTING
     cate_vs_modifier_slope with a known/near-constant propensity (the
     within-session empirical stim-vs-control trial fraction; assignment is
     an experimenter-set pseudo-randomised sequence, not something to
     estimate).

Y = trial correct (1) / error (0) -- read directly from the correct/ vs
error/ file split (no per-trial flag needed). This is the behavioral read-
out; WP-CAUSAL-2's other modifiers (flow-divergence phase, dynamic/stable
subspace) and the Soldado-language dynamic-vs-stable replication (WP-CAUSAL-5)
are deferred to Phase 2 once the gate result is in.

Outputs:
  results/causal_soldado.json -- per-session features + the pooled CATE-vs-
    alignment slope test (the gate).
  results/all_statistics.json -- "causal_soldado" key.

Run:
    conda run -n wm_dynamics python scripts/run_soldado_pipeline.py
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import pca_decompose
from dynamics import dmd_reconstruction_error
from control import controllability_gramian, average_controllability, modal_controllability
from causal import cate_vs_modifier_slope, benchmark_modifiers
from statistics import stable_seed, paired_sign_flip_test
from io_utils import locked_json_update

DATA = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory"
    "/data/Soldado-Magraner/PFC_uStim_data/data"
)
RESULTS = ROOT / "results"

# STEP B-EXT: optional TRUE structural-controllability competitor (Markov &
# Kennedy 2014 macaque FLNe connectome). Absent -> M2a/M2b/M6 are skipped
# cleanly; the base benchmark (M1-M5) is unaffected.
CONNECTOME_PATH = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory"
    "/data/connectomes/markov2014_fln.csv"
)
# The Soldado-Magraner uStim arrays sit over the dorsolateral principal-
# sulcus PFC; area-level assignment only (no per-electrode histology in this
# dataset -- Methods states this limitation), same area for every session.
DLPFC_MARKOV_AREA = "9/46d"
N_RANDOM_DIRS = 20


def _load_connectome() -> tuple[np.ndarray, list[str]] | tuple[None, None]:
    if not CONNECTOME_PATH.exists():
        print(f"structural competitor skipped (no connectome at {CONNECTOME_PATH})")
        return None, None
    with open(CONNECTOME_PATH) as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    areas = lines[0].strip().split(",")
    W = np.loadtxt(lines[1:], delimiter=",")
    return W, areas

BIN_S = 0.05          # spikerate bin size (s), fixed by the dataset
PRE_S = 0.8            # every trial's spikerate starts at -0.8 s re: stim onset
N_BINS = 30             # common post-crop window: -0.8 .. +0.7 s. Retains 100% of trials
                        # (the global per-trial minimum across all 11 sessions is exactly
                        # 30 bins) and covers more of the post-stim delay period than the
                        # original 26-bin window, while staying close to the ~28-bin
                        # (~1.4s) window used in the paper's own PNBdecoder stats
                        # (stats/decoder/*_PNBdecoder.mat, accuracy shape (28, n_cond)).
N_PC = 8
DMD_RANK = 7   # Round-7 STEP A: CV-selected primary rank (results/dmd_rank_selection.json
               # _meta.selected_r=7); r in {5,6,7,8} robustness lives in that same file's
               # per-r benchmark_slope/benchmark_p (run_dmd_rank_selection.py's
               # benchmark_slope_at_rank, which flips this constant per-rank on this module).
GRAMIAN_HORIZON = 20

SESSIONS = [
    "Sa210311_s224",
    "Wa220801_s549", "Wa220802_s550", "Wa220803_s551", "Wa220804_s552",
    "Wa220805_s553", "Wa220808_s554", "Wa220809_s555", "Wa220810_s556",
    "Wa220811_s557", "Wa220812_s558",
]


# ── Format-specific loaders, normalised to one common representation ──────────
#
# Common representation returned by both loaders:
#   dict:
#     trials       : list of dict(spikerate=(T,C) float32 [time, channel],
#                                  stim_cond=int, angle_idx=int)
#     stim_channels: list[str | None] -- per stim_cond, the raw uStim channel
#                     token(s) as decoded from the file (None = control)
#     channel_ids  : (C,) int -- physical electrode ID for each spikerate column
#     control_idx  : int -- index into stim_channels/spikerate-cond-axis that is
#                    the no-stim control condition

def _ascii_to_str(arr: np.ndarray) -> str:
    return "".join(chr(int(x)) for x in np.asarray(arr).ravel())


def _parse_chan_token(token: str) -> list[int]:
    """'265  281' -> [265, 281]; '0  0' / '0' -> []; '31' -> [31]."""
    nums = [int(x) for x in re.findall(r"\d+", token)]
    return [n for n in nums if n != 0]


def _load_h5py_session(mat_path: Path) -> dict | None:
    import h5py

    try:
        f = h5py.File(str(mat_path), "r")
    except OSError:
        return None

    with f:
        params = f["params"]
        channel_ids = f["params"]["channels"][()][0].astype(int)  # (C,)

        cond_chan_tokens = []
        cc = params["cond_uStimChan"]
        for i in range(cc.shape[0]):
            cond_chan_tokens.append(_ascii_to_str(f[cc[i, 0]][()]))
        n_angle = params["cond_targetAngle"].shape[0]

        sr = f["spikerate"]
        n_stim_cond = sr.shape[3]  # h5py-reversed: (trial, angle, amp, uStimCond)
        trials = []
        for c in range(n_stim_cond):
            for a in range(n_angle):
                for tr in range(sr.shape[0]):
                    ref = sr[tr, a, 0, c]
                    cell = f[ref]
                    if cell.shape is None or len(cell.shape) != 2 or cell.shape[0] < 2:
                        continue
                    mat = cell[()].astype(np.float32)  # (time, channel) already
                    if mat.shape[1] != len(channel_ids):
                        continue
                    trials.append({"spikerate": mat, "stim_cond": c, "angle_idx": a})

        control_idx = next(
            (i for i, tok in enumerate(cond_chan_tokens) if not _parse_chan_token(tok)),
            None,
        )
        return {
            "trials": trials,
            "stim_channels": [_parse_chan_token(t) for t in cond_chan_tokens],
            "channel_ids": channel_ids,
            "control_idx": control_idx,
        }


def _load_scipy_session(mat_path: Path) -> dict | None:
    import scipy.io as sio

    if not mat_path.exists():
        return None
    d = sio.loadmat(str(mat_path), simplify_cells=True)
    params = d["params"]
    channel_ids = np.asarray(params["channels"])[:, 0].astype(int)

    cond_chan_tokens = list(np.atleast_1d(params["cond_uStimChan"]))
    n_stim_cond, n_angle = d["spikerate"].shape[0], d["spikerate"].shape[1]

    trials = []
    sr = d["spikerate"]
    for c in range(n_stim_cond):
        for a in range(n_angle):
            cell_list = sr[c, a]
            if isinstance(cell_list, np.ndarray) and cell_list.dtype != object:
                cell_list = [cell_list]
            for cell in np.atleast_1d(cell_list):
                if not isinstance(cell, np.ndarray) or cell.ndim != 2 or cell.shape[1] < 2:
                    continue
                mat = cell.T.astype(np.float32)  # (channel, time) -> (time, channel)
                if mat.shape[1] != len(channel_ids):
                    continue
                trials.append({"spikerate": mat, "stim_cond": c, "angle_idx": a})

    control_idx = next(
        (i for i, tok in enumerate(cond_chan_tokens) if not _parse_chan_token(str(tok))),
        None,
    )
    return {
        "trials": trials,
        "stim_channels": [_parse_chan_token(str(t)) for t in cond_chan_tokens],
        "channel_ids": channel_ids,
        "control_idx": control_idx,
    }


def load_soldado_session(prefix: str, correct: bool) -> dict | None:
    """Load one session's correct-only or error-only trial pool.

    MAT-file generation (v7.3/HDF5 vs v5/v7) varies per FILE, not per monkey
    or even per session -- e.g. a session's correct-trial file can be plain
    v7 while its error-trial file is v7.3. Detect by trying to open as HDF5
    first (cheap: h5py raises OSError immediately on a non-HDF5 signature)
    rather than trusting the filename.
    """
    import h5py

    folder = "correct" if correct else "error"
    fname = f"{prefix}.mat" if correct else f"{prefix}_err.mat"
    mat_path = DATA / folder / fname
    if not mat_path.exists():
        return None
    try:
        with h5py.File(str(mat_path), "r"):
            is_h5 = True
    except OSError:
        is_h5 = False
    return _load_h5py_session(mat_path) if is_h5 else _load_scipy_session(mat_path)


# ── Geometric feature extraction (all numerics reused from src/) ──────────────

def crop_trial(mat: np.ndarray) -> np.ndarray | None:
    """(T,C) -> (N_BINS,C), or None if the trial is shorter than the window."""
    if mat.shape[0] < N_BINS:
        return None
    return mat[:N_BINS]


def build_session_features(prefix: str, structural_ctrl: dict | None = None) -> dict | None:
    corr = load_soldado_session(prefix, correct=True)
    if corr is None or corr["control_idx"] is None:
        return None
    err = load_soldado_session(prefix, correct=False)

    control_idx = corr["control_idx"]
    channel_ids = corr["channel_ids"]
    C = len(channel_ids)

    def _epochs_for(cond_source: dict, cond: int, label_correct: int) -> list[tuple[np.ndarray, int, int]]:
        out = []
        for tr in cond_source["trials"]:
            if tr["stim_cond"] != cond:
                continue
            cropped = crop_trial(tr["spikerate"])
            if cropped is not None:
                out.append((cropped, label_correct, tr["angle_idx"]))
        return out

    # 1. PCA + DMD fit on CONTROL, CORRECT trials only.
    ctrl_epochs = [e for e, _, _ in _epochs_for(corr, control_idx, 1)]
    if len(ctrl_epochs) < 10:
        return None
    Z_ctrl = np.stack(ctrl_epochs, axis=0)  # (N, N_BINS, C)
    X_flat = Z_ctrl.reshape(-1, C)
    _, V, var_ratio = pca_decompose(X_flat, N_PC)  # V: (C, k)
    k = V.shape[1]

    Z_ctrl_mean = ((Z_ctrl.reshape(-1, C) - X_flat.mean(0)) @ V).reshape(Z_ctrl.shape[0], N_BINS, k).mean(0)
    dt = BIN_S
    r_use = min(DMD_RANK, k, N_BINS - 2)
    dmd = dmd_reconstruction_error(Z_ctrl_mean, r=r_use, dt=dt)
    A = dmd["A"]
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = vecs[:, order[0]].real
    v_star = v_star / (np.linalg.norm(v_star) + 1e-12)

    # Round-11 Part 17B: w1, the top RIGHT SINGULAR vector of the SAME
    # control-epoch-fit A -- the one-step maximal-gain input direction for a
    # non-normal plant (see scripts/run_amplification_check.py Part 17A,
    # median cos(v*,w1)=0.36 across sessions -- the two directions genuinely
    # differ here, so both are scored as competing causal targets below).
    _, _, _Vt_svd = np.linalg.svd(A)
    w1 = _Vt_svd[0].real
    w1 = w1 / (np.linalg.norm(w1) + 1e-12)

    # Round-11 Part 19: P_k, the k-dim WM-manifold projector in CHANNEL space
    # (V is the same control-epoch PCA loading used to fit A above) -- for the
    # Sadtler et al. 2014 within/outside-manifold constraint, scored on the
    # raw channel-space stim direction (see run_manifold_constraint.py Part 19A).
    Pk = V @ V.T  # (C, C)

    # M3 reference direction: the SLOW/stable (context-like) mode -- smallest
    # |eigenvalue| -- content-specificity control (steering it should not
    # scale the causal effect the way steering v_star does).
    stable_order = np.argsort(np.abs(eigs))
    v_stable = vecs[:, stable_order[0]].real
    v_stable = v_stable / (np.linalg.norm(v_stable) + 1e-12)

    # M4 no-structure null floor: fixed-seed random unit directions in the
    # SAME latent space, independent of the fitted dynamics.
    rand_rng = np.random.default_rng(stable_seed(f"soldado_random_dirs_{prefix}"))
    random_dirs = rand_rng.standard_normal((N_RANDOM_DIRS, k))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-12

    # 1b. WP-CAUSAL-5: dynamic-vs-stable subspace split (Soldado-Magraner et al.'s
    # own result, re-derived in this project's language): classify each latent
    # eigendirection of A as DYNAMIC (large rotational/imaginary component -- the
    # rotating-memorandum analogue) or STABLE (near the real axis -- the
    # fixed-context analogue), then ask whether stimulation perturbs the dynamic
    # subspace more than the stable one, exactly the R1/R2 content/context
    # dissociation applied causally.
    im_mag = np.abs(eigs.imag)
    order_im = np.argsort(im_mag)[::-1]
    half = max(1, k // 2)
    dyn_idx, stab_idx = order_im[:half], order_im[half:]

    def _subspace_activation(traj_latent: np.ndarray, idx: np.ndarray) -> float:
        # traj_latent: (T, k) real latent trajectory; idx: eigenvector indices.
        proj = np.abs(vecs[:, idx].conj().T @ traj_latent.T)  # (len(idx), T)
        return float(proj.mean())

    # 2. Per stim-condition B, alignment-to-v*, Gramian trace.
    n_cond = len(corr["stim_channels"])
    cond_features = {}
    for c in range(n_cond):
        if c == control_idx:
            continue
        chan_ids = corr["stim_channels"][c]
        idx = [i for i, cid in enumerate(channel_ids) if cid in chan_ids]
        if not idx:
            continue
        B_chan = np.zeros((C, 1))
        B_chan[idx, 0] = 1.0 / len(idx)
        B_lat = V.T @ B_chan  # (k, 1)
        B_hat_unit = B_lat[:, 0] / (np.linalg.norm(B_lat) + 1e-12)
        align = float(np.abs(B_hat_unit @ v_star))
        Wc = controllability_gramian(A, B_lat, T=GRAMIAN_HORIZON)

        # M6: the Gramian's TOP eigenvector -- the control-OPTIMAL steering
        # direction -- distinct from v_star (the dynamics' native unstable mode).
        wc_eigvals, wc_eigvecs = np.linalg.eigh(Wc)
        min_energy_dir = wc_eigvecs[:, np.argmax(wc_eigvals)]
        min_energy_dir_alignment = float(np.abs(B_hat_unit @ min_energy_dir))

        # M7 (Round-11 Part 17B): alignment to w1, the propagator's top
        # singular vector -- the amplification-reframe competitor to v_star.
        amplification_alignment = float(np.abs(B_hat_unit @ w1))

        # M8 (Round-11 Part 19): within/outside-manifold fractions of the raw
        # channel-space stim direction B_chan (== e_hat in run_manifold_
        # constraint.py's Part 19A, same P_k), NOT B_lat (which is
        # within-manifold by construction -> would be degenerate/circular).
        e_hat = B_chan[:, 0]
        e_norm = float(np.linalg.norm(e_hat)) + 1e-12
        within_frac = float(np.linalg.norm(Pk @ e_hat) / e_norm)
        outside_frac = float(np.linalg.norm(e_hat - Pk @ e_hat) / e_norm)

        cond_features[c] = {
            "alignment_to_vstar": align, "gramian_trace": float(np.trace(Wc)),
            "stim_channels": chan_ids,
            "stable_alignment": float(np.abs(B_hat_unit @ v_stable)),
            "random_alignment": float(np.mean(np.abs(random_dirs @ B_hat_unit))),
            "input_norm": float(np.linalg.norm(B_lat)),
            "min_energy_dir_alignment": min_energy_dir_alignment,
            "amplification_alignment": amplification_alignment,
            "within_frac": within_frac,
            "outside_frac": outside_frac,
        }
        if structural_ctrl is not None:
            cond_features[c]["anat_avg_ctrl"] = structural_ctrl["anat_avg_ctrl"]
            cond_features[c]["anat_modal_ctrl"] = structural_ctrl["anat_modal_ctrl"]

    # 2b. WP-CAUSAL-5 test: mean dynamic- and stable-subspace activation, control
    # vs (all conditions pooled) stimulated trials.
    all_ctrl_epochs = [e for e, _, _ in _epochs_for(corr, control_idx, 1)] + (
        [e for e, _, _ in _epochs_for(err, control_idx, 0)] if err is not None else [])
    all_stim_epochs = []
    for c in cond_features:
        all_stim_epochs += [e for e, _, _ in _epochs_for(corr, c, 1)]
        if err is not None:
            all_stim_epochs += [e for e, _, _ in _epochs_for(err, c, 0)]

    def _latent(ep: np.ndarray) -> np.ndarray:
        return (ep - X_flat.mean(0)) @ V

    dyn_stim_test = None
    if len(all_ctrl_epochs) >= 10 and len(all_stim_epochs) >= 10:
        dyn_ctrl_vals = np.array([_subspace_activation(_latent(e), dyn_idx) for e in all_ctrl_epochs])
        stab_ctrl_vals = np.array([_subspace_activation(_latent(e), stab_idx) for e in all_ctrl_epochs])
        dyn_stim_vals = np.array([_subspace_activation(_latent(e), dyn_idx) for e in all_stim_epochs])
        stab_stim_vals = np.array([_subspace_activation(_latent(e), stab_idx) for e in all_stim_epochs])
        dyn_pct_change = float((dyn_stim_vals.mean() - dyn_ctrl_vals.mean()) / (dyn_ctrl_vals.mean() + 1e-12))
        stab_pct_change = float((stab_stim_vals.mean() - stab_ctrl_vals.mean()) / (stab_ctrl_vals.mean() + 1e-12))
        dyn_stim_test = {
            "dynamic_pct_change": dyn_pct_change,
            "stable_pct_change": stab_pct_change,
            "abs_dynamic_minus_abs_stable": abs(dyn_pct_change) - abs(stab_pct_change),
        }

    # 3. Trial-level table: for each stim condition, stim trials (t=1) + the
    #    shared control pool (t=0), both correct- and error-file trials, all
    #    tagged with that condition's modifier. Confounders X are PRE-TREATMENT
    #    covariates only (target angle, session) -- not gramian_trace, which is
    #    derived from the same (A, B, v*) geometry as the modifier itself and
    #    would risk collider-style circularity in the nuisance models.
    modifier_keys = [k for k in ("gramian_trace", "stable_alignment", "random_alignment",
                                 "input_norm", "min_energy_dir_alignment", "amplification_alignment",
                                 "within_frac", "outside_frac",
                                 "anat_avg_ctrl", "anat_modal_ctrl") if k in next(iter(cond_features.values()), {})]

    rows = []
    ctrl_all = _epochs_for(corr, control_idx, 1) + (
        _epochs_for(err, control_idx, 0) if err is not None else []
    )
    for c, feat in cond_features.items():
        stim_all = _epochs_for(corr, c, 1) + (_epochs_for(err, c, 0) if err is not None else [])
        if len(stim_all) < 5 or len(ctrl_all) < 5:
            continue
        n_stim, n_ctrl = len(stim_all), len(ctrl_all)
        propensity = n_stim / (n_stim + n_ctrl)
        extra = {mk: feat[mk] for mk in modifier_keys}
        for _, y, angle_idx in stim_all:
            rows.append({"y": y, "t": 1, "modifier": feat["alignment_to_vstar"],
                         "propensity": propensity, "angle_idx": angle_idx, **extra})
        for _, y, angle_idx in ctrl_all:
            rows.append({"y": y, "t": 0, "modifier": feat["alignment_to_vstar"],
                         "propensity": propensity, "angle_idx": angle_idx, **extra})

    return {
        "session": prefix,
        "n_control_trials": len(ctrl_all),
        "n_pc": k,
        "var_explained": float(var_ratio.sum()),
        "max_real_eig": float(eigs[order[0]].real),
        "cond_features": {str(c): v for c, v in cond_features.items()},
        "dynamic_vs_stable": dyn_stim_test,
        "rows": rows,
    }


def main():
    W_connectome, connectome_areas = _load_connectome()
    structural_ctrl = None
    if W_connectome is not None:
        avg_ctrl = average_controllability(W_connectome)
        modal_ctrl = modal_controllability(W_connectome)
        area_i = connectome_areas.index(DLPFC_MARKOV_AREA)
        structural_ctrl = {"anat_avg_ctrl": float(avg_ctrl[area_i]),
                           "anat_modal_ctrl": float(modal_ctrl[area_i])}
        print(f"Structural competitor: area-level proxy at Markov area "
              f"'{DLPFC_MARKOV_AREA}' (same for every session) -- "
              f"avg_ctrl={structural_ctrl['anat_avg_ctrl']:.4f}, "
              f"modal_ctrl={structural_ctrl['anat_modal_ctrl']:.4f}")

    all_rows = []
    per_session = {}
    session_order = []
    for prefix in SESSIONS:
        print(f"  {prefix} ...", end=" ")
        try:
            feat = build_session_features(prefix, structural_ctrl=structural_ctrl)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if feat is None:
            print("SKIP (insufficient data)")
            continue
        print(f"{len(feat['rows'])} rows, {len(feat['cond_features'])} stim conditions, "
              f"var_explained={feat['var_explained']:.2f}, max_real_eig={feat['max_real_eig']:.3f}")
        per_session[prefix] = {k: v for k, v in feat.items() if k != "rows"}
        session_idx = len(session_order)
        session_order.append(prefix)
        for r in feat["rows"]:
            r["session_idx"] = session_idx
        all_rows.extend(feat["rows"])

    if not all_rows:
        print("No usable Soldado sessions -- stopping without a gate result.")
        return

    y = np.array([r["y"] for r in all_rows], dtype=float)
    t = np.array([r["t"] for r in all_rows], dtype=int)
    modifier = np.array([r["modifier"] for r in all_rows], dtype=float)
    propensity = np.array([r["propensity"] for r in all_rows], dtype=float)

    # X = pre-treatment confounders only: target-angle one-hot + session one-hot.
    angle_idx = np.array([r["angle_idx"] for r in all_rows], dtype=int)
    session_idx = np.array([r["session_idx"] for r in all_rows], dtype=int)
    angle_oh = np.eye(angle_idx.max() + 1)[angle_idx]
    session_oh = np.eye(len(session_order))[session_idx]
    X = np.hstack([angle_oh, session_oh])

    print(f"\nPooled: N={len(y)} rows ({int(t.sum())} stim, {int((1 - t).sum())} control), "
          f"accuracy stim={y[t == 1].mean():.3f} control={y[t == 0].mean():.3f}")

    rng = np.random.default_rng(stable_seed("soldado_causal_gate"))
    gate = cate_vs_modifier_slope(
        y, t, X, modifier=modifier, propensity=propensity, n_perm=5000, rng=rng,
    )
    print(f"\nGATE: slope={gate['slope']:.4f} [{gate['slope_ci_lo']:.4f}, {gate['slope_ci_hi']:.4f}] "
          f"p={gate['p_value']:.4f} (ATE={gate['ate']:.4f}, N={gate['n']})")
    verdict = "PASS" if (gate["slope"] > 0 and gate["p_value"] < 0.05) else "FAIL/WEAK"
    print(f"GATE VERDICT: {verdict}")

    # STEP B: the benchmark -- v* (M1) adjudicated against every competing
    # theory of the causally-relevant controllable direction, on the SAME
    # cross-fit pseudo-outcome. Modifiers absent from a session's rows (i.e.
    # never computed, e.g. the structural columns if no connectome) are
    # simply absent from `all_rows`' keys -- guard with .get.
    def _col(key):
        return np.array([r.get(key, np.nan) for r in all_rows], dtype=float)

    # Round-10 Part 15A: trial-resolution dissociation arm. v_star itself is
    # already fit ONCE per session (control-epoch DMD, see build_session_
    # features above) -- there is no per-trial v* VECTOR at this call site to
    # average, so per spec 15A's own fallback: collapse the existing per-
    # condition scalar `vstar_alignment` to its session mean and broadcast
    # back to every row in that session. This is the fixed, pre-computable
    # target a clinician could use without re-estimating v* every trial --
    # the static competitor to trial/condition-resolved vstar_alignment.
    session_mean_vstar_scalar = np.empty_like(modifier)
    for s in range(len(session_order)):
        mask = session_idx == s
        session_mean_vstar_scalar[mask] = np.nanmean(modifier[mask])

    modifiers = {
        "vstar_alignment": modifier,
        "gramian_trace": _col("gramian_trace"),
        "stable_alignment": _col("stable_alignment"),
        "random_alignment": _col("random_alignment"),
        "input_norm": _col("input_norm"),
        "min_energy_dir_alignment": _col("min_energy_dir_alignment"),
        "amplification_alignment": _col("amplification_alignment"),
        "session_mean_vstar_scalar": session_mean_vstar_scalar,
    }
    # anat_avg_ctrl / anat_modal_ctrl are area-level structural-controllability
    # scalars (same value for every session -- see the print above), i.e. ZERO
    # within-arena variance. They are not a per-trial steering direction and
    # cannot be scored as a rankable competitor here; benchmark_modifiers'
    # zero-variance guard would exclude them anyway, so don't even pass them in
    # (Round-8.1 Part 11A). They remain in structural_ctrl/cond_features for
    # the Boran-side benchmark and as a descriptive note.

    bench_rng = np.random.default_rng(stable_seed("soldado_benchmark"))
    bench = benchmark_modifiers(y, t, X, modifiers=modifiers, propensity=propensity,
                                n_perm=5000, rng=bench_rng)
    print(f"\nBENCHMARK leaderboard (winner={bench['winner']}, N={bench['n']}):")
    for name, row in sorted(bench["leaderboard"].items(), key=lambda kv: -kv[1]["slope"]):
        print(f"  {name:26s} slope={row['slope']:+.4f} "
              f"[{row['slope_ci_lo']:+.4f}, {row['slope_ci_hi']:+.4f}] p={row['p_value']:.4f}")
    print("BENCHMARK joint model:")
    for name, row in bench["joint"].items():
        print(f"  {name:26s} coef={row['coef']:+.4f} p={row['p_value']:.4f}")
    print("BENCHMARK nested (dR2 over the winner alone):")
    for name, row in bench["nested"].items():
        print(f"  {name:26s} dR2={row['dR2']:+.5f} p={row['p_value']:.4f}")
    if bench["winner"] != "vstar_alignment":
        print(f"HONESTY: the benchmark winner is '{bench['winner']}', NOT vstar_alignment -- "
              f"report this plainly, it is the finding.")

    # M1 marginal SUPERSETS the existing single-modifier gate: unstandardize
    # the z-scored benchmark slope (z = (m-mean)/sd -> raw = z-slope / SD(modifier))
    # and compare to the gate's raw-scale slope -- an independent cross-fit
    # (its own rng), so agreement is only expected up to cross-fit sampling
    # noise, not bit-for-bit.
    vstar_sd = float(np.nanstd(modifier))
    m1_raw_recovered = bench["leaderboard"]["vstar_alignment"]["slope"] / vstar_sd
    m1_matches_gate = abs(m1_raw_recovered - gate["slope"]) < 0.25 * abs(gate["slope"])
    print(f"\nM1-vs-gate check: gate slope={gate['slope']:.4f}; benchmark M1 "
          f"unstandardized={m1_raw_recovered:.4f} "
          f"({'MATCH' if m1_matches_gate else 'MISMATCH -- reporting as-is, see comments.txt STEP B2'})")

    bench_json = {"leaderboard": bench["leaderboard"], "joint": bench["joint"],
                  "nested": bench["nested"], "winner": bench["winner"], "n": bench["n"],
                  "structural_competitor_used": structural_ctrl is not None,
                  "dlpfc_markov_area": DLPFC_MARKOV_AREA if structural_ctrl is not None else None,
                  "m1_vs_gate": {"gate_slope": gate["slope"], "m1_unstandardized_slope": m1_raw_recovered,
                                "matches": m1_matches_gate},
                  "dissociation_arm": "session_mean_vstar_scalar",
                  "dissociation_arm_note": ("FALLBACK operationalization (Part 15A): v_star is fit once "
                                            "per session from control-epoch DMD (no per-trial v* vector "
                                            "exists at this call site), so this arm is the session-groupby "
                                            "mean of the per-condition vstar_alignment scalar, broadcast "
                                            "to every row in that session -- the fixed, pre-computable "
                                            "competitor to condition/trial-resolved vstar_alignment."),
                  "excluded": bench.get("excluded", {})}
    with open(RESULTS / "causal_benchmark.json", "w") as f:
        json.dump(bench_json, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    np.savez_compressed(
        RESULTS / "causal_benchmark_detail.npz",
        phi=bench["phi"], **{f"z_{k}": v for k, v in bench["z_modifiers"].items()},
    )

    # WP-CAUSAL-5: pooled test, across sessions, that stimulation perturbs the
    # dynamic subspace more than the stable one (paired by session, sign-flip test
    # on the per-session |dynamic %change| - |stable %change| contrast).
    dvs = [v["dynamic_vs_stable"] for v in per_session.values() if v.get("dynamic_vs_stable") is not None]
    dynamic_vs_stable_result = None
    if len(dvs) >= 4:
        abs_dyn = np.array([abs(d["dynamic_pct_change"]) for d in dvs])
        abs_stab = np.array([abs(d["stable_pct_change"]) for d in dvs])
        res5 = paired_sign_flip_test(abs_dyn, abs_stab, n_perm=10000, alternative="greater",
                                     rng=np.random.default_rng(stable_seed("soldado_wp5")))
        dynamic_vs_stable_result = {
            "n_sessions": len(dvs), "mean_abs_dynamic_pct_change": float(abs_dyn.mean()),
            "mean_abs_stable_pct_change": float(abs_stab.mean()),
            "mean_diff": res5["mean_diff"], "ci_lower": res5["ci_lower"],
            "ci_upper": res5["ci_upper"], "p_value": res5["p_value"],
        }
        print(f"\nWP-CAUSAL-5 (dynamic vs stable perturbation, N={len(dvs)} sessions): "
              f"|dynamic|={abs_dyn.mean():.3f} vs |stable|={abs_stab.mean():.3f}, "
              f"diff={res5['mean_diff']:.3f} [{res5['ci_lower']:.3f}, {res5['ci_upper']:.3f}] "
              f"p={res5['p_value']:.4f}")

    # Figure-support artifact only (Fig 4 CATE-vs-alignment panel): the pseudo-
    # outcome/modifier/permutation-null arrays are not JSON-serializable and
    # are not themselves a new statistical claim (slope/CI/p above already are)
    # -- saved separately so generate_paper_figures.py can plot the actual gate
    # scatter + null instead of a schematic.
    np.savez_compressed(
        RESULTS / "causal_soldado_gate_detail.npz",
        phi=gate["phi"], modifier=gate["modifier"], null=gate["null"],
    )
    gate_json = {k: v for k, v in gate.items() if k not in ("phi", "modifier", "null")}

    out = {
        "per_session": per_session,
        "n_sessions_used": len(per_session),
        "pooled_n": len(all_rows),
        "gate": gate_json,
        "gate_verdict": verdict,
        "dynamic_vs_stable_replication": dynamic_vs_stable_result,
    }
    with open(RESULTS / "causal_soldado.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["causal_soldado"] = out
        stats["causal_benchmark"] = bench_json
    print("\nSaved results/causal_soldado.json, results/causal_soldado_gate_detail.npz, "
          "results/causal_benchmark.json, results/causal_benchmark_detail.npz, "
          "updated all_statistics.json")


if __name__ == "__main__":
    main()
