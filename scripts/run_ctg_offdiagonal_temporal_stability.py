#!/usr/bin/env python3
"""Reads the off-diagonal cells of every results/*_ctg_*.npz cross-temporal
generalisation matrix -- previously unread by any analysis in this project,
which consumed only the diagonal -- and asks whether a decoder fit at one
timepoint within its matrix's own epoch generalises to other timepoints in
that same epoch, gated on that epoch's diagonal being itself decodable so a
weak-decodability epoch cannot be misread as a rotating one.

No on-disk matrix in this project trains at an encoding-epoch timepoint and
tests at a maintenance-epoch timepoint (or vice versa) in the same matrix:
every generating pipeline fits each square matrix over timepoints drawn from
one a-priori window only. Where a corpus also carries a second, independently
fit matrix over a different window (an encoding, baseline, or transient
control), both are read here as separate, clearly labelled cells -- never
combined into a single cross-epoch statistic, because that statistic does
not exist in any artifact this project has computed.

Resampling unit is the subject (sessions of the same subject are averaged
first, never resampled independently), and the multiple-comparison family is
declared in two parts: the six corpora's primary within-epoch window, and
the smaller set of secondary control windows, FDR-corrected separately since
they answer a different question from the primary one.

Outputs: results/ctg_offdiagonal_temporal_stability.json

Run:
    conda run -n wm_dynamics python scripts/run_ctg_offdiagonal_temporal_stability.py
"""
import sys
import re
import json
import inspect
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from geometry import temporal_stability_tau  # noqa: E402
from statistics import Z_80_POWER, fdr_bh, stable_seed  # noqa: E402
from provenance import _json_safe  # noqa: E402
from subject_independence import count_independent_groups, resolve_group  # noqa: E402

RESULTS = ROOT / "results"
N_BOOT = 10000
DIAG_INTERPRETABLE_MIN = inspect.signature(temporal_stability_tau).parameters["min_diag_auc"].default

SUB_RE = re.compile(r"sub-[0-9A-Za-z]+")

CORPORA = {
    "boran": {
        "pattern": "boran_ctg_*.npz",
        "window_description": "maintenance delay, 0-2.8s from maintenance onset",
        "controls": [
            ("auc_mat_encoding_control", "encoding_epoch"),
            ("auc_mat_baseline_control", "baseline_epoch"),
        ],
        "overlap_note": (
            "is a high-gamma field potential view of the same DANDI 000574 "
            "patients as dandi000574_units (sub-01..sub-09 in both); its "
            "corpus-level statistic here is not independent of "
            "dandi000574_units's and the two must never be pooled or added "
            "as separate evidence"
        ),
    },
    "dandi000469": {
        "pattern": "dandi000469_ctg_*.npz",
        "window_description": "maintenance delay, 0-2.3s from maintenance onset",
        "controls": [("auc_mat_encoding_control", "encoding_epoch")],
    },
    "dandi000574_units": {
        "pattern": "dandi000574_units_ctg_*.npz",
        "window_description": "maintenance delay, 0-3.0s from maintenance onset",
        "controls": [],
        "overlap_note": (
            "is a single-unit view of the same DANDI 000574 patients as "
            "boran (sub-01..sub-09 in both); its corpus-level statistic "
            "here is not independent of boran's and the two must never be "
            "pooled or added as separate evidence"
        ),
    },
    "dandi000673": {
        "pattern": "dandi000673_ctg_*.npz",
        "window_description": "maintenance delay, 0-2.3s from maintenance onset",
        "controls": [],
        "overlap_note": (
            "shares roughly 31 of its subjects with dandi001187 across 37 "
            "sessions (dandi001187 is this project's canonical view of "
            "those shared patient-sessions); its corpus-level statistic "
            "here is not independent of dandi001187's and the two must "
            "never be pooled or added as separate evidence"
        ),
    },
    "dandi001187": {
        "pattern": "dandi001187_ctg_*.npz",
        "window_description": "maintenance delay, 0-2.3s from maintenance onset",
        "controls": [],
        "overlap_note": (
            "shares roughly 31 of its subjects with dandi000673 across 37 "
            "sessions; this corpus is this project's canonical view of "
            "those shared patient-sessions, so dandi000673's corpus-level "
            "statistic must never be pooled with or added to this one"
        ),
    },
    "miller": {
        "pattern": "miller_ctg_*.npz",
        "window_description": "0-back/2-back maintenance window, 0.3-1.4s post-cue",
        "controls": [("auc_mat_transient_control", "transient_epoch")],
    },
}

GROUP_DESCRIPTIONS = {
    frozenset(["dandi000469"]): (
        "dandi000469",
        "DANDI 000469 patients, single view",
    ),
    frozenset(["boran", "dandi000574_units"]): (
        "dandi_000574",
        "same DANDI 000574 patients (sub-01..sub-09), two recording "
        "views: boran is high-gamma field potential, "
        "dandi000574_units is single-unit",
    ),
    frozenset(["dandi000673", "dandi001187"]): (
        "dandi000673_dandi001187",
        "same shared patient-sessions, two release views: dandi000673 "
        "and dandi001187 (dandi001187 is this project's canonical view "
        "of those patient-sessions)",
    ),
    frozenset(["miller"]): (
        "miller",
        "electrocorticography subjects of the miller corpus, single view",
    ),
}


def _build_independent_patient_groups() -> dict:
    resolved: dict[str, list] = {}
    for cell in CORPORA:
        resolved.setdefault(resolve_group(cell), []).append(cell)
    membership = {frozenset(cells): cells for cells in resolved.values()}
    groups = {}
    for cells_key, (out_key, description) in GROUP_DESCRIPTIONS.items():
        if cells_key not in membership:
            raise ValueError(
                f"subject_independence_registry.json membership no longer "
                f"matches the expected group {sorted(cells_key)}; update "
                "GROUP_DESCRIPTIONS or investigate the registry change"
            )
        groups[out_key] = {"cells": sorted(cells_key), "description": description}
    if count_independent_groups(list(CORPORA)) != len(groups):
        raise ValueError("independent-group count from the registry disagrees with GROUP_DESCRIPTIONS")
    return groups


INDEPENDENT_PATIENT_GROUPS = _build_independent_patient_groups()


def _independent_group_summary(primary: dict) -> list:
    summary = []
    for group, spec in INDEPENDENT_PATIENT_GROUPS.items():
        cells = spec["cells"]
        verdicts = {c: primary[c]["verdict"] for c in cells}
        if len(cells) == 1:
            note = f"only view of this group, verdict {verdicts[cells[0]]}"
        elif all(v == "clear_stable" for v in verdicts.values()):
            note = "both views clear"
        elif not any(v == "clear_stable" for v in verdicts.values()):
            note = "neither view clears"
        else:
            clearing = [c for c, v in verdicts.items() if v == "clear_stable"]
            gated = [c for c, v in verdicts.items() if v != "clear_stable"]
            note = (
                f"one view clears ({', '.join(clearing)}) and the other "
                f"view of the same patients ({', '.join(gated)}) gates "
                "inconclusive on its own diagonal decodability; different "
                "recording views of the same patients disagreeing is not "
                "a contradiction, the diagonal gate is measuring exactly "
                "that decodability difference"
            )
            if group == "dandi000673_dandi001187":
                note += (
                    ". This project's canonical view of these patients is "
                    "dandi001187, which is the view that gates "
                    "inconclusive here; this tension between the clearing "
                    "view and the canonical view is recorded, not "
                    "resolved, and is left for a human to adjudicate"
                )
        summary.append({
            "group": group,
            "cells": cells,
            "verdicts": verdicts,
            "note": note,
        })
    return summary


def _subject_id(path: Path) -> str:
    stem = path.stem
    match = SUB_RE.search(stem)
    if match:
        return match.group()
    return stem.rsplit("_", 1)[-1]


def _cluster_bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict:
    n = len(values)
    observed = float(np.mean(values))
    boot = np.array([np.mean(values[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
    lower, upper = np.percentile(boot, [2.5, 97.5])
    p_value = float(min(1.0, 2 * min(np.mean(boot <= 0), np.mean(boot >= 0))))
    p_value = max(p_value, 1.0 / (N_BOOT + 1))
    sd = float(np.std(values, ddof=1)) if n > 1 else float("nan")
    mdd = float(Z_80_POWER * sd / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "mean_offdiag_effect": observed,
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "p_value_bootstrap": p_value,
        "mdd": mdd,
        "n_subjects": n,
    }


def _window_analysis(pattern: str, field: str, seed_key: str) -> dict:
    files = sorted(RESULTS.glob(pattern))
    rng = np.random.default_rng(stable_seed(seed_key))
    per_subject: dict[str, list[dict]] = {}
    n_infeasible = 0
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if field not in d.files:
                n_infeasible += 1
                continue
            mat = d[field]
            if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] == 0:
                n_infeasible += 1
                continue
            info = temporal_stability_tau(mat)
        per_subject.setdefault(_subject_id(f), []).append(info)

    n_sessions = sum(len(v) for v in per_subject.values())
    result = {
        "n_files": len(files),
        "n_sessions_used": n_sessions,
        "n_infeasible": n_infeasible,
        "n_subjects": len(per_subject),
    }
    if not per_subject:
        result["status"] = "infeasible"
        return result

    subj_ids = sorted(per_subject)
    offdiag_vals = np.array([
        np.mean([r["offdiag_effect"] for r in per_subject[s]]) for s in subj_ids
    ])
    diag_auc_vals = np.array([
        np.mean([r["mean_diag_auc"] for r in per_subject[s]]) for s in subj_ids
    ])
    tau_vals = np.array([
        np.mean([r["tau"] for r in per_subject[s]]) for s in subj_ids
    ])

    boot = _cluster_bootstrap(offdiag_vals, rng)
    mean_diag_auc = float(np.mean(diag_auc_vals))
    result.update({
        "status": "computed",
        **boot,
        "mean_diag_auc": mean_diag_auc,
        "mean_tau": float(np.nanmean(tau_vals)),
        "diag_interpretable": bool(mean_diag_auc >= DIAG_INTERPRETABLE_MIN),
        "diagonal_below_chance": bool(mean_diag_auc < 0.5),
    })
    return result


def _verdict(stats: dict) -> str:
    if stats.get("status") != "computed":
        return "infeasible"
    if not stats["diag_interpretable"]:
        return "inconclusive_undecodable_diagonal"
    clears = stats.get("q_value_fdr", 1.0) < 0.05 and stats["ci95_lower"] > 0.0
    return "clear_stable" if clears else "inconclusive"


def _reading(stats: dict) -> dict:
    return {
        "offdiag_effect": stats["mean_offdiag_effect"],
        "ci95_lower": stats["ci95_lower"],
        "ci95_upper": stats["ci95_upper"],
        "q_value_fdr": stats["q_value_fdr"],
        "mean_diag_auc": stats["mean_diag_auc"],
        "n_subjects": stats["n_subjects"],
    }


def _apply_fdr(cells: dict) -> None:
    keys = [k for k, v in cells.items() if v["stats"]["status"] == "computed"]
    if not keys:
        return
    p_vals = np.array([cells[k]["stats"]["p_value_bootstrap"] for k in keys])
    fdr = fdr_bh(p_vals, alpha=0.05)
    for k, q in zip(keys, fdr["q_values"]):
        cells[k]["stats"]["q_value_fdr"] = float(q)
    for k in keys:
        cells[k]["verdict"] = _verdict(cells[k]["stats"])
        cells[k]["reading"] = _reading(cells[k]["stats"])
    for k, v in cells.items():
        if k not in keys:
            v["verdict"] = _verdict(v["stats"])


def main():
    primary = {}
    secondary = {}
    for corpus, spec in CORPORA.items():
        stats = _window_analysis(spec["pattern"], "auc_mat", corpus)
        entry = {
            "corpus": corpus,
            "tier": "human",
            "window": "primary",
            "window_description": spec["window_description"],
            "stats": stats,
        }
        if "overlap_note" in spec:
            entry["overlap_note"] = spec["overlap_note"]
        primary[corpus] = entry

        for field, label in spec["controls"]:
            stats_c = _window_analysis(spec["pattern"], field, f"{corpus}_{label}")
            secondary[f"{corpus}_{label}"] = {
                "corpus": corpus,
                "tier": "human",
                "window": label,
                "stats": stats_c,
            }

    _apply_fdr(primary)
    _apply_fdr(secondary)

    n_clear = sum(1 for v in primary.values() if v["verdict"] == "clear_stable")
    n_inconclusive = sum(1 for v in primary.values() if v["verdict"].startswith("inconclusive"))
    n_infeasible = sum(1 for v in primary.values() if v["verdict"] == "infeasible")
    cells_offdiag_interval_excludes_zero = sorted(
        corpus for corpus, entry in primary.items()
        if entry["stats"]["status"] == "computed"
        and (entry["stats"]["ci95_lower"] > 0.0 or entry["stats"]["ci95_upper"] < 0.0)
    )

    out = {
        "description": (
            "Cluster-bootstrap and FDR analysis of off-diagonal cells in "
            "every results/*_ctg_*.npz cross-temporal generalisation matrix "
            "in this project. Statistic is offdiag(AUC-0.5) minus zero, "
            "gated on the same matrix's diagonal decodability via "
            "temporal_stability_tau (src/geometry.py); resampling unit is "
            "the subject. No matrix in this project spans an encoding-to-"
            "maintenance boundary -- see window_description per corpus and "
            "boundary_crossing_gap below."
        ),
        "boundary_crossing_gap": (
            "every generating pipeline fits each square matrix over "
            "timepoints drawn from one a-priori epoch only; a corpus that "
            "also carries a second matrix over a different epoch (an "
            "encoding, baseline, or transient control) stores it as an "
            "independently fit square matrix, never jointly with the "
            "primary window's timepoints, so no on-disk cell trains at one "
            "epoch and tests at another"
        ),
        "trial_disjointness_verdict": (
            "verified from src/geometry.py: ctg_nested_cv fits PCA and the "
            "per-timepoint classifier using only the current fold's "
            "training-trial indices (_fit_pca_fold, clf.fit in "
            "_ctg_score_fold); every entry of the resulting matrix, "
            "diagonal and off-diagonal alike, is then scored on that same "
            "fold's held-out test-trial indices, which are disjoint from "
            "the training indices by construction of the k-fold split in "
            "_ctg_splits. All six generating scripts route through this "
            "function, directly or via load_vs_load_ctg/item_identity_ctg "
            "in src/spike_pipeline.py, which themselves call it."
        ),
        "predeclared_rules": {
            "statistic": "offdiag(AUC-0.5), mean over off-diagonal cells of temporal_stability_tau(auc_mat)",
            "decodability_gate": f"corpus-mean diagonal AUC must be >= {DIAG_INTERPRETABLE_MIN} for the corpus cell to be interpretable at all; below that, verdict is inconclusive_undecodable_diagonal regardless of the off-diagonal value",
            "stability_criterion": "clear_stable requires FDR q < 0.05 AND the 95% cluster-bootstrap CI on mean offdiag(AUC-0.5) excludes zero, both required",
            "resampling_unit": "subject (sessions of the same subject averaged first, never resampled independently)",
            "multiple_comparison_family": "two families, FDR-corrected separately: primary (one cell per corpus, six corpora) and secondary (control-window cells: encoding/baseline/transient, four cells)",
            "power_z_80pct": Z_80_POWER,
            "mdd_formula": "z_80pct * sd(subject-level offdiag effects, ddof=1) / sqrt(n_subjects)",
            "non_clearing_cells": "reported as inconclusive, never as a null -- there is no named reference on the AUC-0.5 scale on this project against which a minimum detectable difference could be judged small",
            "rotation_claims": "this analysis never asserts rotation was detected or refuted; a non-clearing cell says the data do not distinguish stability from rotation, nothing more",
        },
        "primary": primary,
        "secondary_controls": secondary,
        "n_clear_stable_primary_cells": n_clear,
        "cells_with_offdiag_interval_excluding_zero": cells_offdiag_interval_excludes_zero,
        "n_inconclusive_primary_cells": n_inconclusive,
        "n_infeasible_primary_cells": n_infeasible,
        "primary_cell_counts_note": (
            "n_clear_stable_primary_cells/n_inconclusive_primary_cells/"
            "n_infeasible_primary_cells above count the six primary cells, "
            "not independent patient groups -- boran/dandi000574_units are "
            "two views of the same DANDI 000574 patients and "
            "dandi000673/dandi001187 are two views of the same shared "
            "patient-sessions, so six cells are only four independent "
            "groups; see independent_patient_groups and "
            "independent_group_summary"
        ),
        "independent_patient_groups": INDEPENDENT_PATIENT_GROUPS,
        "independent_group_summary": _independent_group_summary(primary),
    }

    with open(RESULTS / "ctg_offdiagonal_temporal_stability.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)

    print(f"clear_stable={n_clear} inconclusive={n_inconclusive} infeasible={n_infeasible}")
    for corpus, entry in primary.items():
        s = entry["stats"]
        if s["status"] == "computed":
            print(f"  {corpus}: offdiag={s['mean_offdiag_effect']:.4f} "
                  f"ci=[{s['ci95_lower']:.4f},{s['ci95_upper']:.4f}] "
                  f"q={s.get('q_value_fdr', float('nan')):.4f} "
                  f"mdd={s['mdd']:.4f} n_subj={s['n_subjects']} "
                  f"verdict={entry['verdict']}")
        else:
            print(f"  {corpus}: {entry['verdict']} (n_files={s['n_files']})")


if __name__ == "__main__":
    main()
