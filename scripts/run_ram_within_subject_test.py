#!/usr/bin/env python3
"""Round-11 PART 20A: within-subject re-test of the RAM causal-targeting
alignment test, gated by run_ram_within_subject_feasibility.py (PASSED:
n_multisite_subjects=12 >= 8 -- see results/causal_ram_within_subject.json).

WHY: run_ram_openloop_pipeline.py's pooled test (results/causal_ram.json,
slope=+0.012, CI[-0.031,0.055], p=0.57, null) builds ONE alignment scalar
per SESSION and pools across subjects with NO subject term -- the graded-
alignment variation there is entirely BETWEEN subjects (confounded by
coverage/pathology/electrode placement). This script refits using ONLY the
WITHIN-subject, across-site variation: for the 12 subjects with >=2 distinct
stim sites (anode/cathode label pairs), the modifier (alignment_to_vstar,
computed identically per session by run_ram_openloop_pipeline.build_session_
features -- NOT redefined here) is demeaned within subject before scoring,
so the slope is identified purely from a subject's own across-site alignment
contrast, not cross-subject confounds. Single-site subjects contribute their
between-session word-level trials to the pooled n (their demeaned modifier
is exactly zero, contributing no across-site information but not biasing the
within-subject estimate either) -- EXCLUDED from the slope-identifying
sample per spec ("within-subject across-site variation only"); we restrict
the refit sample to the 12 multi-site subjects' rows only, the literal set
the within-subject contrast is estimable from.

Cluster-robust bootstrap OVER SUBJECTS (not trial-level), reusing
cate_vs_modifier_slope (src/causal.py untouched).

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_ram_within_subject_test.py
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

from causal import cate_vs_modifier_slope
from statistics import stable_seed
import run_ram_openloop_pipeline as ram  # noqa: E402

RESULTS = ROOT / "results"
N_BOOT = 2000


def main() -> None:
    with open(RESULTS / "causal_ram_within_subject.json") as f:
        gate = json.load(f)
    if gate.get("status") != "feasible_gate_only" or gate.get("n_multisite_subjects", 0) < 8:
        print("Feasibility gate did not pass (or was not run this session) -- stopping without a refit.")
        return
    multisite_subjects = set(gate["multisite_subjects"].keys())
    print(f"Multi-site subjects (n={len(multisite_subjects)}): {sorted(multisite_subjects)}")

    sessions = ram._find_stim_sessions()
    print(f"Found {len(sessions)} candidate bipolar+stim session JSONs; "
          f"rebuilding features exactly as run_ram_openloop_pipeline.build_session_features does ...")

    all_rows, per_session = [], {}
    for ieeg_json in sessions:
        subj = ieeg_json.parts[-4]
        if subj not in multisite_subjects:
            continue
        try:
            feat = ram.build_session_features(ieeg_json)
        except Exception as e:
            print(f"  {ieeg_json.relative_to(ram.DATA)} FAILED: {e}")
            continue
        if feat is None:
            continue
        per_session[feat["session"]] = {k: v for k, v in feat.items() if k != "rows"}
        for r in feat["rows"]:
            r["subject"] = subj
            r["session_stim_channel"] = feat["stim_channel"]
        all_rows.extend(feat["rows"])
        print(f"  {feat['session']} ({subj}): {feat['n_words']} words, "
              f"align={feat['alignment_to_vstar']:.3f}, stim_ch={feat['stim_channel']}")

    subjects_scored = sorted({r["subject"] for r in all_rows})
    subjects_with_ge2_sessions_scored = {
        s for s in subjects_scored
        if len({r["session_stim_channel"] for r in all_rows if r["subject"] == s}) >= 2
    }
    print(f"\nSubjects with usable (quality-filter-passing) sessions: {len(subjects_scored)}")
    print(f"Of those, subjects with >=2 DISTINCT usable stim channels after quality filtering: "
          f"{len(subjects_with_ge2_sessions_scored)} -- {sorted(subjects_with_ge2_sessions_scored)}")

    if len(subjects_with_ge2_sessions_scored) < 8:
        out = {
            "status": "underpowered_after_quality_filtering",
            "n_multisite_subjects_raw_labels": len(multisite_subjects),
            "n_multisite_subjects_after_quality_filter": len(subjects_with_ge2_sessions_scored),
            "reason": (f"raw anode/cathode label parsing found {len(multisite_subjects)} multi-site "
                      f"subjects, but only {len(subjects_with_ge2_sessions_scored)} retain >=2 distinct "
                      "stim channels after run_ram_openloop_pipeline's own quality filters "
                      "(MIN_WORDS, EDF channel-name match, control-epoch count) -- below the n=8 "
                      "feasibility threshold once restricted to analyzable sessions."),
        }
        with open(RESULTS / "causal_ram_within_subject.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nGATE FAILS AFTER QUALITY FILTERING ({len(subjects_with_ge2_sessions_scored)} < 8) "
              "-- stopping, keeping the pooled null as the honest bound.")
        return

    # restrict to the subjects that retain across-site contrast after quality filtering
    rows = [r for r in all_rows if r["subject"] in subjects_with_ge2_sessions_scored]
    y = np.array([r["y"] for r in rows], dtype=float)
    t = np.array([r["t"] for r in rows], dtype=int)
    modifier_raw = np.array([r["modifier"] for r in rows], dtype=float)
    propensity = np.array([r["propensity"] for r in rows], dtype=float)
    subject_arr = np.array([r["subject"] for r in rows])
    X = np.array([[r["serialpos"], r["list"]] for r in rows], dtype=float)

    # within-subject demeaning of the modifier (spec's second option): the
    # slope is then identified purely from each subject's own across-site
    # alignment contrast, not cross-subject confounds.
    modifier_demeaned = modifier_raw.copy()
    for s in subjects_with_ge2_sessions_scored:
        mask = subject_arr == s
        modifier_demeaned[mask] = modifier_raw[mask] - modifier_raw[mask].mean()

    print(f"\nRefit sample: N={len(rows)} rows across {len(subjects_with_ge2_sessions_scored)} "
          f"within-subject-contrast subjects "
          f"({int(t.sum())} stim, {int((1 - t).sum())} control)")

    rng = np.random.default_rng(stable_seed("ram_within_subject_test"))
    within_result = cate_vs_modifier_slope(
        y, t, X, modifier=modifier_demeaned, propensity=propensity, n_perm=5000, rng=rng,
    )
    print(f"Within-subject (demeaned modifier) slope: {within_result['slope']:.4f} "
          f"[{within_result['slope_ci_lo']:.4f}, {within_result['slope_ci_hi']:.4f}] "
          f"p={within_result['p_value']:.4f} (trial-level, ATE={within_result['ate']:.4f}, "
          f"N={within_result['n']})")

    # cluster-robust bootstrap OVER SUBJECTS (not trial-level): resample the
    # within-subject-contrast subjects with replacement, refit the same OLS
    # slope formula on the (already cross-fit) pseudo-outcome each draw.
    phi = within_result["phi"]
    mod = within_result["modifier"]
    # cate_vs_modifier_slope drops non-finite rows before returning phi/modifier
    # -- re-derive the finite-row subject labels the same way (finite in phi &
    # modifier) so cluster indices stay aligned; phi is finite by construction
    # of aipw_pseudo_outcome on this well-behaved binary/propensity design, so
    # the only source of dropped rows here is modifier_demeaned.
    assert len(phi) <= len(rows), "cate_vs_modifier_slope returned more rows than the input"
    finite_mod = np.isfinite(modifier_demeaned)
    if finite_mod.sum() == len(phi):
        subj_for_phi = subject_arr[finite_mod]
    else:
        raise RuntimeError(
            f"finite-row count mismatch: {finite_mod.sum()} finite modifier rows vs "
            f"{len(phi)} returned phi rows -- cannot safely align subject labels for the "
            "cluster bootstrap; stopping rather than risk misalignment."
        )

    def _slope_formula(m: np.ndarray, p: np.ndarray) -> float:
        mc = m - m.mean()
        denom = (mc ** 2).sum()
        if denom < 1e-15:
            return 0.0
        return float((mc * (p - p.mean())).sum() / denom)

    subj_list = sorted(subjects_with_ge2_sessions_scored)
    rows_by_subject = [np.where(subj_for_phi == s)[0] for s in subj_list]
    boot_rng = np.random.default_rng(stable_seed("ram_within_subject_clusterboot"))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        drawn = boot_rng.integers(0, len(subj_list), size=len(subj_list))
        idx = np.concatenate([rows_by_subject[i] for i in drawn]) if all(len(rows_by_subject[i]) for i in drawn) else np.array([], dtype=int)
        if len(idx) == 0:
            boot[b] = 0.0
            continue
        boot[b] = _slope_formula(mod[idx], phi[idx])

    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    point_slope = _slope_formula(mod, phi)
    p_cluster = min(2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean())), 1.0)

    print(f"\nCluster-robust (subject-resampled, B={N_BOOT}): slope={point_slope:+.4f} "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}] p={p_cluster:.4f}")

    with open(RESULTS / "causal_ram.json") as f:
        pooled = json.load(f)["result"]

    is_significant_positive = ci_lo > 0 and p_cluster < 0.05
    if is_significant_positive:
        tail = "a genuinely less-confounded positive signal."
    else:
        tail = (f"as expected for a low-powered ({len(subj_list)}-subject) within-subject contrast, "
               "this null is honestly reported, not spun; it bounds the human causal claim more "
               f"defensibly than the confounded pooled test (also null, slope={pooled['slope']:+.4f}, "
               f"p={pooled['p_value']:.3f}).")
    verdict = (
        f"Within-subject refit (n={len(subj_list)} multi-site subjects, N={len(rows)} word-trials): "
        f"cluster-robust slope={point_slope:+.4f}, 95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}], p={p_cluster:.4f}. "
        f"{'Positive and significant' if is_significant_positive else 'Null'} -- {tail} "
        "The encoding/episodic paradigm mismatch with WM-delay stimulation remains and applies here too."
    )
    print(f"\nVERDICT: {verdict}")

    out = {
        "status": "ran",
        "n_multisite_subjects": len(subj_list),
        "multisite_subjects": subj_list,
        "n_rows": len(rows),
        "within_subject_slope": point_slope,
        "cluster_ci": [float(ci_lo), float(ci_hi)],
        "cluster_p": float(p_cluster),
        "trial_level_slope_for_reference": within_result["slope"],
        "trial_level_p_for_reference": within_result["p_value"],
        "pooled_slope_for_reference": pooled["slope"],
        "pooled_ci_for_reference": [pooled["slope_ci_lo"], pooled["slope_ci_hi"]],
        "pooled_p_for_reference": pooled["p_value"],
        "verdict": verdict,
    }
    with open(RESULTS / "causal_ram_within_subject.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/causal_ram_within_subject.json")


if __name__ == "__main__":
    main()
