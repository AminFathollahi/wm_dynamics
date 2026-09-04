#!/usr/bin/env python3
"""OpenNeuro ds005557 -- "Free Recall with Closed-Loop Stimulation at Encoding
(Encoding Classifier)" (RAM).

Human iEEG, delayed free recall, CLOSED-LOOP (online classifier-triggered,
not experimenter-scheduled) electrical stimulation of encoding-period word
presentations. Same BIDS field conventions and task design as ds005489 (the
open-loop RAM arm, scripts/run_ram_openloop_pipeline.py): WORD events carry
`stimulation` and `recalled` directly, the stimulated bipolar pair is named
in `anode_label`/`cathode_label`, and one stim site is used per subject. The
only substantive difference is WHEN a trial is stimulated: the classifier
decides in real time, based on the subject's own encoding-period neural
state, rather than a fixed experimental schedule. This is the "closed-loop
arm" this project's dataset list calls for alongside the existing open-loop
one; the same v*-alignment causal test is run on it, unchanged, so the two
are directly comparable.

Reuses scripts/run_ram_openloop_pipeline.py's build_session_features
directly (parameterized by data_root) rather than duplicating the epoching /
PCA / DMD / alignment logic -- the two pipelines differ only in which BIDS
dataset they point at.

SCOPE BOUNDARY: as with the open-loop arm, stimulation here is delivered at
ENCODING, not during a WM maintenance/delay period -- state this wherever
these results are reported.

Outputs:
  results/causal_ram_closedloop.json
  results/all_statistics.json -- "causal_ram_closedloop" key.

Run:
    conda run -n wm_dynamics python scripts/run_ram_closedloop_pipeline.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from causal import cate_vs_modifier_slope  # noqa: E402
from statistics import stable_seed  # noqa: E402
from io_utils import locked_json_update  # noqa: E402

from run_ram_openloop_pipeline import build_session_features  # noqa: E402

DATA = dataset_path("ram_ds005557_closedloop")
RESULTS = ROOT / "results"
MAX_SUBJECTS = 38


def _find_stim_sessions() -> list[Path]:
    return sorted(DATA.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))


def main():
    sessions = _find_stim_sessions()
    print(f"Found {len(sessions)} candidate bipolar+stim sessions (ds005557, closed-loop)")

    all_rows, per_session = [], {}
    n_subjects_done = set()
    for ieeg_json in sessions:
        subj = ieeg_json.parts[-4]
        if len(n_subjects_done) >= MAX_SUBJECTS and subj not in n_subjects_done:
            continue
        print(f"  {ieeg_json.relative_to(DATA)} ...", end=" ")
        try:
            feat = build_session_features(ieeg_json, data_root=DATA, derive_stim_from_stim_on=True)
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
        print("No usable ds005557 sessions -- stopping without a result.")
        return

    y = np.array([r["y"] for r in all_rows], dtype=float)
    t = np.array([r["t"] for r in all_rows], dtype=int)
    modifier = np.array([r["modifier"] for r in all_rows], dtype=float)
    propensity = np.array([r["propensity"] for r in all_rows], dtype=float)
    X = np.array([[r["serialpos"], r["list"]] for r in all_rows], dtype=float)

    print(f"\nPooled: N={len(y)} rows ({int(t.sum())} stim, {int((1 - t).sum())} control) "
          f"across {len(per_session)} sessions, recall stim={y[t == 1].mean():.3f} "
          f"ctrl={y[t == 0].mean():.3f}")

    rng = np.random.default_rng(stable_seed("ram_closedloop_causal_test"))
    result = cate_vs_modifier_slope(
        y, t, X, modifier=modifier, propensity=propensity, n_perm=5000, rng=rng,
    )
    print(f"\nRESULT: slope={result['slope']:.4f} "
          f"[{result['slope_ci_lo']:.4f}, {result['slope_ci_hi']:.4f}] "
          f"p={result['p_value']:.4f} (ATE={result['ate']:.4f}, N={result['n']})")

    np.savez_compressed(RESULTS / "causal_ram_closedloop_detail.npz",
                        phi=result["phi"], modifier=result["modifier"], null=result["null"])
    result_json = {k: v for k, v in result.items() if k not in ("phi", "modifier", "null")}

    out = {
        "per_session": per_session,
        "n_sessions_used": len(per_session),
        "pooled_n": len(all_rows),
        "result": result_json,
    }
    with open(RESULTS / "causal_ram_closedloop.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)

    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["causal_ram_closedloop"] = out
    print("\nSaved results/causal_ram_closedloop.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
