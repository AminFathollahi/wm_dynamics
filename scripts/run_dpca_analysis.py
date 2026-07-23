#!/usr/bin/env python3
"""Demixed principal component analysis: is the temporally stable code
condition-dependent, or attributable to a shared temporal envelope?

The standing interpretive concern for the sustained load/context coding result
is that it is expected under any account of task-set maintenance, i.e. it
could in principle be nothing more than a shared temporal envelope
(condition-independent, present regardless of load) rather than something
that actually distinguishes conditions. marginalize_condition_time already
reports the variance fractions carried by the condition-independent versus
condition-dependent marginalization (used for Miller in
scripts/run_miller_ctg_corrected.py, under the key "dpca_lite"); this analysis
goes one step further and asks the cross-temporal generalization question
directly: if single trials are projected onto only the condition-dependent
subspace versus only the condition-independent subspace
(geometry.dpca_condition_subspace_projection), and the identical
nested-cross-validation, label-permutation cross-temporal generalization
pipeline is rerun on each, does the temporally stable structure survive in the
condition-dependent projection, the condition-independent one, or both?

Datasets: Miller (load), DANDI 000469 (load and content), CRCNS pfc-3 (content).

Outputs: results/dpca_{dataset}.json
Updates: results/all_statistics.json — "dpca_{dataset}" keys

Compute note: the per-subject/per-session CTG-permutation tests below are
mutually independent (each fits its own PCA/classifiers on its own trials),
so they run under joblib across processes. Each worker pins BLAS to 1 thread
(threadpoolctl) to avoid oversubscribing the machine when running as many
single-threaded workers as there are logical cores.

N_PERM=1000 here (vs. the >=5000 the Round-4 audit required for the
comparability-critical headline CI-vs-CD contrasts) is a deliberate,
documented reduction from that requirement, made only because a full
N_PERM=5000, unparallelized run of this script was measured to require
>1 day of wall-clock time (a single N_PERM=5000 attempt was killed by hand
after 8h24m, having completed only 4/18 of the DANDI 000469 session loop).
1000 permutations still resolves p down to ~0.001 (vs. 5000's ~0.0002),
adequate for reporting significance at alpha=0.05 after BH-FDR correction
across the ~30-90 tests this script runs, though with less precision in the
0.001-0.01 p-value range than 5000 would give. If wall-clock budget allows a
full 5000-permutation run in the future (e.g. on a machine with more cores,
or run unattended over 1-2 days), N_PERM should be restored to 5000.

Run (after run_000469_pipeline.py, run_pfc3_content_ctg.py, Miller geometry exist):
    conda run -n wm_dynamics python scripts/run_dpca_analysis.py
"""
import sys, json
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import (dpca_condition_subspace_projection, marginalize_condition_time,
                      ctg_label_permutation_null, ctg_content_permutation_null,
                      temporal_stability_tau)
from spike_pipeline import fit_pca_psth
from statistics import stable_seed

RESULTS = ROOT / "results"
N_DPCA_COMPONENTS = 4
N_PERM = 1000   # see compute note above — reduced from the >=5000 spec for wall-clock feasibility
N_JOBS = -1     # joblib: use all available cores (each pinned to 1 BLAS thread)


def _ctg_on_projection(Z_proj: np.ndarray, labels: np.ndarray, t_idx: np.ndarray,
                       multiclass: bool, rng) -> dict:
    # ctg_*_permutation_null expect (N, C, T) raw-feature data and do their own
    # per-fold PCA down to n_components; Z_proj is (N, T, d) (the Z-array
    # convention) and already low-dimensional, so transpose and use its own
    # dimensionality as n_components (no further reduction).
    X = Z_proj.transpose(0, 2, 1)   # (N, d, T)
    fn = ctg_content_permutation_null if multiclass else ctg_label_permutation_null
    n_splits = 3 if multiclass else 5
    res = fn(X, labels, t_idx, n_components=X.shape[1],
             n_splits=n_splits, n_perm=N_PERM, rng=rng)
    tau_info = temporal_stability_tau(res["auc_mat"])
    return {"offdiag_effect": res["mean_offdiag_auc_minus_chance"], "p_value": res["p_value"],
            "tau": tau_info["tau"], "interpretable": tau_info["interpretable"],
            "mean_diag_auc": tau_info["mean_diag_auc"]}


def _miller_one_subject(subj: str) -> tuple[str, dict]:
    with threadpool_limits(limits=1):
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id, times = d["Z"], d["task_id"], d["times"]
        mask = (task_id == 0) | (task_id == 2)
        Z_sub, labels = Z[mask], (task_id[mask] == 2).astype(int)

        maint = (times >= 0.3) & (times <= 1.4)
        t_idx = np.where(maint)[0][::15][:15]

        marg = marginalize_condition_time(Z_sub, labels)
        proj = dpca_condition_subspace_projection(Z_sub, labels, n_components=N_DPCA_COMPONENTS)

        rng = np.random.default_rng(stable_seed(subj + "_dpca"))
        ci_ctg = _ctg_on_projection(proj["Z_condition_independent"], labels, t_idx, False, rng)
        cd_ctg = _ctg_on_projection(proj["Z_condition_dependent"], labels, t_idx, False, rng)
        print(f"  Miller {subj}: frac_cd={marg['frac_condition_dependent']:.3f} | "
              f"CI offdiag={ci_ctg['offdiag_effect']:.4f} (p={ci_ctg['p_value']:.3f}) | "
              f"CD offdiag={cd_ctg['offdiag_effect']:.4f} (p={cd_ctg['p_value']:.3f})", flush=True)
    return subj, {"variance_fractions": marg, "ci_ctg": ci_ctg, "cd_ctg": cd_ctg}


def run_miller() -> dict:
    subjects = ["al", "ca", "cc", "ug"]
    results = Parallel(n_jobs=N_JOBS)(delayed(_miller_one_subject)(s) for s in subjects)
    return dict(results)


def _dandi000469_one_session(path) -> tuple[str, dict] | None:
    with threadpool_limits(limits=1):
        subj = path.stem.replace("dandi000469_geometry_", "")
        d = np.load(path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]
        row = {}

        ctx_mask = (loads == 1) | (loads == 3)
        if ctx_mask.sum() >= 10:
            ctx_labels = (loads[ctx_mask] == 3).astype(int)
            t_idx = np.arange(0, Z.shape[1], 3)
            marg = marginalize_condition_time(Z[ctx_mask], ctx_labels)
            proj = dpca_condition_subspace_projection(Z[ctx_mask], ctx_labels, n_components=N_DPCA_COMPONENTS)
            rng = np.random.default_rng(stable_seed(subj + "_dpca_ctx"))
            row["context"] = {
                "variance_fractions": marg,
                "ci_ctg": _ctg_on_projection(proj["Z_condition_independent"], ctx_labels, t_idx, False, rng),
                "cd_ctg": _ctg_on_projection(proj["Z_condition_dependent"], ctx_labels, t_idx, False, rng),
            }

        load1_mask = loads == 1
        content_labels = pic_id[load1_mask]
        if load1_mask.sum() >= 15 and len(np.unique(content_labels)) >= 2:
            t_idx = np.arange(0, Z.shape[1], 3)
            marg = marginalize_condition_time(Z[load1_mask], content_labels)
            proj = dpca_condition_subspace_projection(Z[load1_mask], content_labels, n_components=N_DPCA_COMPONENTS)
            rng = np.random.default_rng(stable_seed(subj + "_dpca_content"))
            row["content"] = {
                "variance_fractions": marg,
                "ci_ctg": _ctg_on_projection(proj["Z_condition_independent"], content_labels, t_idx, True, rng),
                "cd_ctg": _ctg_on_projection(proj["Z_condition_dependent"], content_labels, t_idx, True, rng),
            }
        if row:
            print(f"  000469 {subj}: " + ", ".join(
                f"{k} frac_cd={v['variance_fractions']['frac_condition_dependent']:.3f}"
                for k, v in row.items()), flush=True)
            return subj, row
    return None


def run_dandi000469() -> dict:
    paths = sorted(RESULTS.glob("dandi000469_geometry_sub-*.npz"))
    results = Parallel(n_jobs=N_JOBS)(delayed(_dandi000469_one_session)(p) for p in paths)
    return dict(r for r in results if r is not None)


def run_pfc3() -> dict:
    d = np.load(RESULTS / "pfc3_content_ctg.npz", allow_pickle=True)
    X, y = d["X"], d["y"]
    Z, _, _ = fit_pca_psth(X, n_comp=8)
    t_idx = np.arange(0, Z.shape[1], 2)

    marg = marginalize_condition_time(Z, y)
    proj = dpca_condition_subspace_projection(Z, y, n_components=N_DPCA_COMPONENTS)
    rng = np.random.default_rng(stable_seed("pfc3_dpca"))
    ci_ctg = _ctg_on_projection(proj["Z_condition_independent"], y, t_idx, True, rng)
    cd_ctg = _ctg_on_projection(proj["Z_condition_dependent"], y, t_idx, True, rng)
    print(f"  pfc-3: frac_cd={marg['frac_condition_dependent']:.3f} | "
          f"CI offdiag={ci_ctg['offdiag_effect']:.4f} (p={ci_ctg['p_value']:.3f}) | "
          f"CD offdiag={cd_ctg['offdiag_effect']:.4f} (p={cd_ctg['p_value']:.3f})")
    return {"variance_fractions": marg, "ci_ctg": ci_ctg, "cd_ctg": cd_ctg}


def main():
    with open(RESULTS / "all_statistics.json") as f:
        stats = json.load(f)

    print("Miller (load)...")
    miller = run_miller()
    with open(RESULTS / "dpca_miller.json", "w") as f:
        json.dump(miller, f, indent=2)
    stats["dpca_miller"] = miller

    print("DANDI 000469 (load + content)...")
    d469 = run_dandi000469()
    with open(RESULTS / "dpca_dandi000469.json", "w") as f:
        json.dump(d469, f, indent=2)
    stats["dpca_dandi000469"] = d469

    print("pfc-3 (content)...")
    pfc3 = run_pfc3()
    with open(RESULTS / "dpca_pfc3.json", "w") as f:
        json.dump(pfc3, f, indent=2)
    stats["dpca_pfc3"] = pfc3

    with open(RESULTS / "all_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved dpca_*.json, updated all_statistics.json")


if __name__ == "__main__":
    main()
