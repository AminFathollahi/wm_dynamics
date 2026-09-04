#!/usr/bin/env python3
"""OpenNeuro ds005489 — "Free Recall with Open-Loop Stimulation at Encoding" (RAM).

Human iEEG, delayed free recall, open-loop (experimenter-randomized, not
decoded-state-triggered) electrical stimulation of alternating two-word
blocks during list encoding. Powered human network-level causal set for
WP-CAUSAL-1/1b/2/3: does the effect of encoding-period stimulation on
later recall grow with the stimulation's geometric alignment to the
unstable eigenvector v* of the encoding-period population dynamics?

SCOPE BOUNDARY: stimulation here is delivered at ENCODING, not during a
WM maintenance/delay period (unlike macaque PFC microstimulation/Rutishauser/Boran). This
dataset therefore speaks to Q4/Q5 (site/actuation dependence) and Q6
(state-vs-behavior) at encoding, not to the delay-period control claim
directly -- state this explicitly wherever these results are reported.

BIDS layout (standard, no format quirks like macaque PFC microstimulation's mixed MAT
generations): sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.edf + *_events.tsv
(fields documented in the sidecar events.json) + *_channels.tsv. The
`stimulation` and `recalled` columns on WORD events already give T and Y
directly -- no manual window-matching against STIM_ON events needed.

Design (mirrors scripts/run_macaque_pfc_microstimulation_pipeline.py's structure exactly; all
numerical work reused from src/geometry.py, src/dynamics.py, src/causal.py):
  1. Per session: epoch high-gamma power around every WORD onset.
  2. Fit PCA + DMD on NON-STIMULATED word epochs only -> v*.
  3. B = one-hot at the session's stimulated bipolar channel (only one
     stim site per subject in this dataset, unlike macaque PFC microstimulation's multi-site
     design) -> alignment-to-v* (one scalar per session).
  4. Y = recalled (0/1) per word, T = stimulation (0/1) per word,
     modifier = that session's alignment (same value for every word in
     the session -- the causal-effect heterogeneity this dataset can
     speak to is BETWEEN-session, via each session's own stim-site
     geometry, pooled with thousands of within-session word-level trials
     for power on the T->Y relationship itself).
  5. Propensity = known design fraction (word-level P(stim)=0.5 within
     stim lists x P(stim list)=n_stim_lists/n_lists; pass the empirical
     per-session fraction in directly, per WP-CAUSAL-1's rule for
     experimentally-assigned treatment).

Outputs:
  results/causal_ram.json
  results/all_statistics.json -- "causal_ram" key.

Run:
    conda run -n wm_dynamics python scripts/run_ram_openloop_pipeline.py
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

from geometry import pca_decompose
from dynamics import dmd_reconstruction_error
from causal import cate_vs_modifier_slope
from statistics import stable_seed
from control import canonicalize_eigenvector_phase
from io_utils import locked_json_update
from preprocessing import high_gamma_power, line_noise_notch

DATA = dataset_path("ram_ds005489_openloop")
RESULTS = ROOT / "results"

PRE_S, POST_S = 0.3, 1.6   # epoch window relative to word onset (word on-screen 1.6s)
BIN_S = 0.1
N_PC = 8
DMD_RANK = 8   # Aligned to the full latent rank (== N_PC), the same cross-cohort-poolability
               # convention used by run_divergence_analysis.py, run_multiband_analysis.py,
               # run_behavior_geometry_link.py, and run_contraction_behavior_analysis_000469.py.
               # Previously 6 with no stated rationale; the epoch window here (19 bins) comfortably
               # supports r=8.
MIN_WORDS = 100          # skip sessions with too few usable WORD epochs
MAX_SUBJECTS = 38        # cap for a full run; set lower for a quick smoke test


def _find_stim_sessions() -> list[Path]:
    return sorted(DATA.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))


def _load_events(events_tsv: Path) -> list[dict]:
    with open(events_tsv) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _derive_word_stimulation(words: list[dict], stim_on: list[dict], stim_off: list[dict],
                             window_s: float = 2.0) -> None:
    """Mutate `words` in place, setting each event's 'stimulation' field from
    STIM_ON/STIM_OFF timestamp overlap rather than trusting the WORD event's
    own field. Some closed-loop RAM releases (e.g. ds005557) record real
    STIM_ON/STIM_OFF events but leave every WORD event's own `stimulation`
    field at "0" -- the online classifier's trigger timing is not backfilled
    onto the word row it applies to. A word is marked stimulated if any
    STIM_ON falls within [onset, onset + window_s] of it (word presentation
    plus a margin for triggering/pulse-train latency), matched to the closest
    such word if a STIM_ON is nearer to more than one.

    Also copies the matched STIM_ON event's own dose fields (amplitude,
    pulse_freq, n_pulses, pulse_width) onto the word row: unlike ds005489,
    this release does not carry those fields on the WORD row itself, only on
    the STIM_ON event that triggered delivery."""
    if not stim_on:
        return
    word_onsets = [float(w["onset"]) for w in words]
    for stim_event in sorted(stim_on, key=lambda s: float(s["onset"])):
        stim_t = float(stim_event["onset"])
        candidates = [(abs(stim_t - w_on), i) for i, w_on in enumerate(word_onsets)
                     if 0 <= stim_t - w_on <= window_s]
        if not candidates:
            continue
        _, best_i = min(candidates)
        words[best_i]["stimulation"] = "1"
        for field in ("amplitude", "pulse_freq", "n_pulses", "pulse_width"):
            if field in stim_event:
                words[best_i][field] = stim_event[field]


def _float_or_nan(value) -> float:
    """Parse a BIDS TSV field (always a string from csv.DictReader) that may
    be a number, 'n/a', or empty -- never raises."""
    try:
        if value in (None, "", "n/a"):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_session_features(ieeg_json: Path, data_root: Path = DATA,
                           derive_stim_from_stim_on: bool = False,
                           return_epochs: bool = False) -> dict | None:
    """Build one session's per-word high-gamma feature set and, when
    `return_epochs` is set, also return the per-trial epoch array and
    channel/dose bookkeeping a downstream analysis needs to work directly
    with the unaveraged (n_words, n_bins, n_channels) log-power epochs
    rather than only the scalar summary this function has always returned.
    Callers that only want the scalar summary (the existing open-loop and
    closed-loop causal pipelines) are unaffected: the extra fields are only
    added when explicitly requested, and nothing already returned changes."""
    import mne

    with open(ieeg_json) as f:
        meta = json.load(f)
    if not meta.get("ElectricalStimulation", False):
        return None

    stem = str(ieeg_json).replace("_ieeg.json", "")
    events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
    edf_path = Path(stem + "_ieeg.edf")
    if not events_tsv.exists() or not edf_path.exists():
        return None

    events = _load_events(events_tsv)
    words = [e for e in events if e["trial_type"] == "WORD"]
    if len(words) < MIN_WORDS:
        return None

    if derive_stim_from_stim_on:
        stim_on = [e for e in events if e["trial_type"] == "STIM_ON"]
        stim_off = [e for e in events if e["trial_type"] == "STIM_OFF"]
        _derive_word_stimulation(words, stim_on, stim_off)
        stim_word = next((w for w in words if w["stimulation"] == "1"), None)
        anode_source = next((s for s in stim_on if s["anode_label"] != "n/a"), None)
        if stim_word is None or anode_source is None:
            return None
        anode, cathode = anode_source["anode_label"], anode_source["cathode_label"]
    else:
        stim_word = next((w for w in words if w["stimulation"] == "1"), None)
        if stim_word is None or stim_word["anode_label"] == "n/a":
            return None
        anode, cathode = stim_word["anode_label"], stim_word["cathode_label"]

    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    ch_names = raw.ch_names
    stim_ch = f"{anode}-{cathode}"
    if stim_ch not in ch_names:
        stim_ch = f"{cathode}-{anode}"
    if stim_ch not in ch_names:
        return None
    srate = raw.info["sfreq"]
    n_ch = len(ch_names)
    rec_dur = raw.times[-1]

    pre_samp = int(PRE_S * srate)
    post_samp = int(POST_S * srate)
    pad = int(0.3 * srate)  # extra padding on each side to absorb Hilbert/filter edge effects
    n_bins = int((PRE_S + POST_S) / BIN_S)

    epochs, recalled, stim_flag, kept_words = [], [], [], []
    for w in words:
        onset_s = float(w["onset"])
        if onset_s - PRE_S - pad / srate < 0 or onset_s + POST_S + pad / srate > rec_dur:
            continue
        i0 = int(round((onset_s - PRE_S) * srate)) - pad
        i1 = int(round((onset_s + POST_S) * srate)) + pad
        raw_seg = raw.get_data(start=i0, stop=i1)  # (n_ch, T_pad)
        raw_seg_notched = line_noise_notch(raw_seg.T, srate, fundamental=60.0, n_harmonics=3)  # US mains
        hgp = high_gamma_power(raw_seg_notched, srate=srate, lo=70.0, hi=150.0, smooth_ms=50.0)
        hgp = hgp[pad:-pad]  # drop padding -> (pre_samp+post_samp, n_ch)
        # bin into BIN_S bins
        n_samp_bin = hgp.shape[0] // n_bins
        binned = hgp[: n_samp_bin * n_bins].reshape(n_bins, n_samp_bin, n_ch).mean(axis=1)
        epochs.append(binned.astype(np.float32))
        recalled.append(int(w["recalled"]))
        stim_flag.append(int(w["stimulation"]))
        kept_words.append(w)

    if len(epochs) < MIN_WORDS:
        return None

    epochs = np.stack(epochs, axis=0)  # (N, n_bins, n_ch)
    recalled = np.array(recalled, dtype=int)
    stim_flag = np.array(stim_flag, dtype=int)

    ctrl_mask = stim_flag == 0
    if ctrl_mask.sum() < 30:
        return None
    epochs_log = np.log1p(np.clip(epochs, 0, None))  # log-power, stabilizes HGP variance
    # Per-channel z-score (stats from control trials only, matching "fit the plant
    # from unperturbed data"): un-normalized channels would otherwise let PCA be
    # dominated by absolute power scale (deep vs. superficial contacts) rather
    # than cross-channel correlation structure.
    mu = epochs_log[ctrl_mask].mean(axis=(0, 1))
    sd = epochs_log[ctrl_mask].std(axis=(0, 1)) + 1e-8
    epochs_z = (epochs_log - mu) / sd
    Z_ctrl = epochs_z[ctrl_mask]
    X_flat = Z_ctrl.reshape(-1, n_ch)
    if not np.all(np.isfinite(X_flat)) or X_flat.std() < 1e-8:
        return None
    _, V, var_ratio = pca_decompose(X_flat, N_PC)
    k = V.shape[1]

    Z_ctrl_mean = ((Z_ctrl.reshape(-1, n_ch) - X_flat.mean(0)) @ V).reshape(
        Z_ctrl.shape[0], epochs.shape[1], k).mean(0)
    dt = BIN_S
    r_use = min(DMD_RANK, k, epochs.shape[1] - 2)
    dmd = dmd_reconstruction_error(Z_ctrl_mean, r=r_use, dt=dt)
    A = dmd["A"]
    eigs, vecs = np.linalg.eig(A)
    order = np.argsort(eigs.real)[::-1]
    v_star = canonicalize_eigenvector_phase(vecs[:, order[0]])

    B_chan = np.zeros((n_ch, 1))
    B_chan[ch_names.index(stim_ch), 0] = 1.0
    B_lat = V.T @ B_chan
    align = float(np.abs((B_lat[:, 0] / (np.linalg.norm(B_lat) + 1e-12)) @ v_star))

    propensity = float(stim_flag.mean())
    rows = [{
        "y": float(recalled[i]), "t": int(stim_flag[i]),
        "modifier": align, "propensity": propensity,
        "serialpos": int(words[i]["serialpos"]) if i < len(words) else -1,
        "list": int(words[i]["list"]) if i < len(words) and words[i]["list"] not in ("-1", "n/a") else -1,
    } for i in range(len(recalled))]

    result = {
        "session": str(edf_path.relative_to(data_root)),
        "n_words": len(recalled),
        "n_pc": k,
        "var_explained": float(var_ratio.sum()),
        "max_real_eig": float(eigs[order[0]].real),
        "stim_channel": stim_ch,
        "alignment_to_vstar": align,
        "propensity": propensity,
        "recall_rate_stim": float(recalled[stim_flag == 1].mean()) if (stim_flag == 1).any() else float("nan"),
        "recall_rate_ctrl": float(recalled[stim_flag == 0].mean()),
        "rows": rows,
    }
    if return_epochs:
        result["epochs_log"] = epochs_log  # (n_words, n_bins, n_ch), pre-z-score log high-gamma power
        result["ch_names"] = list(ch_names)
        result["anode"] = anode
        result["cathode"] = cathode
        result["stim_flag"] = stim_flag
        result["recalled"] = recalled
        # kept_words is aligned index-for-index with epochs/recalled/stim_flag (both loops above
        # skip the same trials, out-of-window ones); indexing the pre-filter `words` list here
        # instead would silently misalign dose values with the trials they were measured on.
        result["amplitude"] = np.array([_float_or_nan(w.get("amplitude")) for w in kept_words], dtype=float)
        result["pulse_freq"] = np.array([_float_or_nan(w.get("pulse_freq")) for w in kept_words], dtype=float)
        result["n_pulses"] = np.array([_float_or_nan(w.get("n_pulses")) for w in kept_words], dtype=float)
        result["pulse_width"] = np.array([_float_or_nan(w.get("pulse_width")) for w in kept_words], dtype=float)
        result["serialpos"] = np.array([int(w.get("serialpos", -1)) for w in kept_words], dtype=int)
        result["list_number"] = np.array(
            [int(w["list"]) if w.get("list") not in ("-1", "n/a", None) else -1 for w in kept_words], dtype=int)
    return result


def main():
    sessions = _find_stim_sessions()
    print(f"Found {len(sessions)} candidate bipolar+stim sessions")

    all_rows, per_session = [], {}
    n_subjects_done = set()
    for ieeg_json in sessions:
        subj = ieeg_json.parts[-4]
        if len(n_subjects_done) >= MAX_SUBJECTS and subj not in n_subjects_done:
            continue
        print(f"  {ieeg_json.relative_to(DATA)} ...", end=" ")
        try:
            feat = build_session_features(ieeg_json)
        except Exception as e:
            print(f"FAILED: {e}")
            continue
        if feat is None:
            print("SKIP (insufficient/unusable data)")
            continue
        n_subjects_done.add(subj)
        print(f"{feat['n_words']} words, align={feat['alignment_to_vstar']:.3f}, "
              f"recall stim={feat['recall_rate_stim']:.3f} ctrl={feat['recall_rate_ctrl']:.3f}, "
              f"var_explained={feat['var_explained']:.2f}, max_real_eig={feat['max_real_eig']:.3f}")
        per_session[feat["session"]] = {k: v for k, v in feat.items() if k != "rows"}
        all_rows.extend(feat["rows"])

    if not all_rows:
        print("No usable ds005489 sessions -- stopping without a result.")
        return

    y = np.array([r["y"] for r in all_rows], dtype=float)
    t = np.array([r["t"] for r in all_rows], dtype=int)
    modifier = np.array([r["modifier"] for r in all_rows], dtype=float)
    propensity = np.array([r["propensity"] for r in all_rows], dtype=float)
    X = np.array([[r["serialpos"], r["list"]] for r in all_rows], dtype=float)

    print(f"\nPooled: N={len(y)} rows ({int(t.sum())} stim, {int((1 - t).sum())} control) "
          f"across {len(per_session)} sessions, recall stim={y[t == 1].mean():.3f} "
          f"ctrl={y[t == 0].mean():.3f}")

    rng = np.random.default_rng(stable_seed("ram_causal_test"))
    result = cate_vs_modifier_slope(
        y, t, X, modifier=modifier, propensity=propensity, n_perm=5000, rng=rng,
    )
    print(f"\nRESULT: slope={result['slope']:.4f} "
          f"[{result['slope_ci_lo']:.4f}, {result['slope_ci_hi']:.4f}] "
          f"p={result['p_value']:.4f} (ATE={result['ate']:.4f}, N={result['n']})")

    # cate_vs_modifier_slope's phi/modifier/null are the pseudo-outcome/permutation-null
    # arrays used internally to compute the scalar fields above -- not a new statistical
    # claim, but not JSON-serializable either (same convention as run_macaque_pfc_microstimulation_pipeline.py's
    # gate_json / causal_macaque_pfc_microstimulation_gate_detail.npz split).
    np.savez_compressed(RESULTS / "causal_ram_detail.npz",
                        phi=result["phi"], modifier=result["modifier"], null=result["null"])
    result_json = {k: v for k, v in result.items() if k not in ("phi", "modifier", "null")}

    out = {
        "per_session": per_session,
        "n_sessions_used": len(per_session),
        "pooled_n": len(all_rows),
        "result": result_json,
    }
    with open(RESULTS / "causal_ram.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["causal_ram"] = out
    print("\nSaved results/causal_ram.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
