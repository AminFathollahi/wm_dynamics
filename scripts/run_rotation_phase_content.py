#!/usr/bin/env python3
"""Content-as-phase test: does the delay-period phase angle in the leading
2-D plane encode item identity?

Gated on the 25D rotation-gate result (results/agent_report.md): a rotating
leading DMD mode is absent in the primary cell for six of seven cohorts, but
the phase of the trajectory's own leading 2-D projection is well-defined
regardless of whether the underlying DMD fit is complex -- dynamics.
ring_attractor_phase (already validated in this paper's existing Rayleigh
test, which found phases CONCENTRATED rather than uniform, itself already
evidence against a clean ring attractor) is reused directly, not duplicated.

Two matched-timepoint tests, at every subsampled delay-window timepoint:
  (i)  a circular multi-group test of phase by item identity
       (statistics.circular_anova_permutation_test);
  (ii) decoding item identity from phase alone ([cos(phi), sin(phi)]) versus
       from the full latent state, same cross-validation folds and
       classifier.
If phase-only decoding tracks full-state decoding, phase carries the content
code. If phase-only sits at chance while the full state decodes well,
content is not phase-encoded in this plane.

Only two cohorts have genuine, decodable item identity in this project:
DANDI 000469 (load-1 trials, 5-way picture identity) and CRCNS pfc-3 (9-way
cue location). All other cohorts either lack item identity altogether or use
near-unique pictures per trial, making identity classification inapplicable.

Aggregation follows the session-level discipline used throughout this
project: session-level circular-test p-values are Stouffer-combined across
sessions at each matched timepoint (trial-level p-values are not reported as
the primary number), and statistics.fdr_bh corrects across timepoints.

Output: results/rotation_phase_content.json.

Run (needs the external data mount):
    conda run -n wm_dynamics python scripts/run_rotation_phase_content.py
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
sys.path.insert(0, str(ROOT / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from dynamics import ring_attractor_phase  # noqa: E402
from statistics import circular_anova_permutation_test, fdr_bh, stable_seed, stouffer_combine  # noqa: E402

import run_axis_rotation_analysis as raa  # noqa: E402  (STEP_000469, STEP_PFC3)

RESULTS = ROOT / "results"
N_PERM = 2000
N_SPLITS = 5
MIN_PER_CLASS = 5


def _phase_at_t(Z: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    N = Z.shape[0]
    phi = np.empty((N, len(t_idx)))
    for n in range(N):
        phi[n] = ring_attractor_phase(Z[n])[t_idx]
    return phi


def _cv_accuracy(X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    min_class = int(np.min(np.bincount(y - y.min())))
    k = max(2, min(N_SPLITS, min_class))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(0, 2**31)))
    accs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(C=1.0, max_iter=1000)
        clf.fit(X[tr], y[tr])
        accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def _run_session(Z: np.ndarray, labels: np.ndarray, t_idx: np.ndarray, rng: np.random.Generator) -> list[dict]:
    phi = _phase_at_t(Z, t_idx)
    rows = []
    for i, t in enumerate(t_idx):
        anova = circular_anova_permutation_test(phi[:, i], labels, n_perm=N_PERM, rng=rng)
        acc_phase = _cv_accuracy(np.stack([np.cos(phi[:, i]), np.sin(phi[:, i])], axis=1), labels, rng)
        acc_full = _cv_accuracy(Z[:, t, :], labels, rng)
        rows.append({"t_idx": int(t), "anova_p": anova["p_value"], "anova_statistic": anova["statistic"],
                     "acc_phase": acc_phase, "acc_full": acc_full, "chance": 1.0 / len(np.unique(labels))})
    return rows


def run_dandi000469() -> dict:
    per_session = {}
    for path in sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz")):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]
        load1_mask = loads == 1
        labels = pic_id[load1_mask]
        if load1_mask.sum() < 15 or len(np.unique(labels)) < 2:
            continue
        counts = np.bincount(labels - labels.min())
        if counts[counts > 0].min() < MIN_PER_CLASS:
            continue
        rng = np.random.default_rng(stable_seed(f"rotation_phase_content_dandi000469_{subj}"))
        t_idx = np.arange(0, Z.shape[1], raa.STEP_000469)
        per_session[subj] = _run_session(Z[load1_mask], labels, t_idx, rng)
    return per_session


def run_pfc3() -> list[dict]:
    d = np.load(RESULTS / "pfc3_content_ctg.npz", allow_pickle=True)
    from spike_pipeline import fit_pca_psth
    Z, _, _ = fit_pca_psth(d["X"], n_comp=8)
    rng = np.random.default_rng(stable_seed("rotation_phase_content_pfc3"))
    t_idx = np.arange(0, Z.shape[1], raa.STEP_PFC3)
    return _run_session(Z, d["y"], t_idx, rng)


def _combine_across_sessions(per_session: dict) -> dict:
    by_t: dict[int, list] = {}
    for rows in per_session.values():
        for row in rows:
            by_t.setdefault(row["t_idx"], []).append(row)
    combined = {}
    for t, rows in sorted(by_t.items()):
        p_vals = np.array([r["anova_p"] for r in rows])
        comb = stouffer_combine(p_vals)
        combined[t] = {
            "p_combined": comb["p_combined"], "n_sessions": len(rows),
            "mean_acc_phase": float(np.mean([r["acc_phase"] for r in rows])),
            "mean_acc_full": float(np.mean([r["acc_full"] for r in rows])),
            "chance": rows[0]["chance"],
        }
    p_by_t = np.array([v["p_combined"] for v in combined.values()])
    fdr = fdr_bh(p_by_t)
    for (t, v), q in zip(combined.items(), fdr["q_values"]):
        v["q_fdr"] = float(q)
    return combined


def _verdict(mean_acc_phase: float, mean_acc_full: float, chance: float, any_q_significant: bool) -> str:
    near_full = mean_acc_phase >= mean_acc_full - 0.03
    near_chance = mean_acc_phase <= chance + 0.05
    if near_full and not near_chance:
        return "phase carries the content code (phase-only decoding tracks full-state decoding)"
    if near_chance and not any_q_significant:
        return ("content is NOT phase-encoded in this plane -- phase-only decoding is near chance "
                "while the circular-mean test finds no reliable item effect either")
    return "mixed: phase-only decoding neither matches full-state nor sits at chance -- reported as-is"


def main():
    print("29 (DANDI 000469, load-1, item identity) ...")
    d469 = run_dandi000469()
    d469_combined = _combine_across_sessions(d469)
    for t, v in d469_combined.items():
        print(f"  t_idx={t}: p_combined={v['p_combined']:.4f} q={v['q_fdr']:.4f} "
              f"acc_phase={v['mean_acc_phase']:.3f} acc_full={v['mean_acc_full']:.3f} "
              f"chance={v['chance']:.3f} (N={v['n_sessions']} sessions)")

    print("\n29 (CRCNS pfc-3, cue location) ...")
    pfc3_rows = run_pfc3()
    for row in pfc3_rows:
        print(f"  t_idx={row['t_idx']}: p={row['anova_p']:.4f} acc_phase={row['acc_phase']:.3f} "
              f"acc_full={row['acc_full']:.3f} chance={row['chance']:.3f}")
    pfc3_fdr = fdr_bh(np.array([r["anova_p"] for r in pfc3_rows]))

    d469_verdict = _verdict(
        float(np.mean([v["mean_acc_phase"] for v in d469_combined.values()])),
        float(np.mean([v["mean_acc_full"] for v in d469_combined.values()])),
        next(iter(d469_combined.values()))["chance"],
        bool(np.any(np.array([v["q_fdr"] for v in d469_combined.values()]) < 0.05)),
    )
    pfc3_verdict = _verdict(
        float(np.mean([r["acc_phase"] for r in pfc3_rows])),
        float(np.mean([r["acc_full"] for r in pfc3_rows])),
        pfc3_rows[0]["chance"], bool(np.any(pfc3_fdr["reject"])),
    )
    print(f"\nDANDI 000469 verdict: {d469_verdict}")
    print(f"pfc-3 verdict: {pfc3_verdict}")

    out = {
        "dandi000469": {"per_session": d469, "combined_by_timepoint": d469_combined,
                        "verdict": d469_verdict, "n_sessions": len(d469)},
        "pfc3": {"rows": pfc3_rows,
                "fdr": {"q_values": pfc3_fdr["q_values"].tolist(), "n_reject": pfc3_fdr["n_reject"]},
                "verdict": pfc3_verdict},
    }
    with open(RESULTS / "rotation_phase_content.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/rotation_phase_content.json")


if __name__ == "__main__":
    main()
