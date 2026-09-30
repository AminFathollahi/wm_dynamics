#!/usr/bin/env python3
"""Region-resolved firing rate, cross-temporal state stability, and behaviour.

Computes, separately per anatomical region rather than pooled across a whole
recording, how three human single-unit quantities relate to one another:

  1. firing rate      -- per-unit mean rate in the maintenance-window delay
                          (build_psth / unit_mean_firing_rates,
                          src/spike_pipeline.py).
  2. state stability   -- within-maintenance-window cross-temporal
                          generalisation off-diagonal effect (ctg_nested_cv +
                          temporal_stability_tau, src/geometry.py): a decoder
                          trained at one delay timepoint and tested at another
                          that still generalises is reading a stable code.
  3. behaviour         -- trial accuracy, error trials admitted (never used
                          to select trials, only as the outcome).

Four human single-unit corpora carry per-unit anatomical labels: dandi_000469,
dandi_001187, dandi_000673, dandi_000574. dandi_000673 and dandi_001187 share
roughly 31 patients across 37 sessions and are one independence group
(provenance/subject_independence_registry.json); dandi_000469 is a second
group; dandi_000574 is a third.

Two confounds are controlled explicitly rather than left to leak into a
regional read:

  - unit count differs about six-fold across regions, so every between-region
    comparison is fit at a common, subsampled unit count (never on the raw
    unmatched counts), reusing the draw-then-pool convention of
    scripts/run_count_subsampling_ladder.py.
  - firing rate is not independent of the population activity level that
    drives decodability, so every rate-linked and stability-linked coupling
    to behaviour is also reported controlling for per-trial population spike
    count (partial_correlation_permutation_test, src/statistics.py), the same
    control this project uses elsewhere against a spike-count confound.

Resampling unit is the patient throughout, never the unit and never the
trial. Every fitted cell carries an estimate, a 95% CI, a minimum detectable
difference, and a q-value as a bare number -- no cell is labelled significant,
cleared, or failing anywhere in this file.

Per-(corpus, region) checkpointing lives under
results/.checkpoints/run_region_resolved_rate_stability_behaviour/, updated
after every session so a restart resumes rather than recomputing.

Outputs: results/region_resolved_rate_stability_behaviour.json

Run:
    conda run -n wm_dynamics python scripts/run_region_resolved_rate_stability_behaviour.py
"""
import sys
import json
from pathlib import Path

import numpy as np
import h5py

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from project_config import dataset_path
from spike_pipeline import (  # noqa: E402
    load_spike_times, build_psth, unit_mean_firing_rates, low_rate_unit_mask,
    resolve_unit_regions, filter_units_by_region, FrozenPSTHTransform,
    ANATOMICAL_REGIONS, BORAN_ANATOMICAL_REGIONS, MIN_UNITS_PER_REGION,
    MIN_UNIT_FIRING_RATE_HZ, MIN_SESSION_ACCURACY, N_PC_DEFAULT,
    BIN_MS_DEFAULT, SMOOTH_MS_DEFAULT,
)
from corpus_sessions import region_filtered_units, MIN_TRIALS  # noqa: E402
from geometry import ctg_nested_cv, temporal_stability_tau  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, stable_seed, fdr_bh, bootstrap_ci, pearson_permutation_test,
    partial_correlation_permutation_test, minimum_detectable_paired_difference,
)
from state_persistence import slope_across_sessions_test  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from subject_independence import resolve_group, count_independent_groups  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from corpus_sessions import pool_draws_within_session
from statistics import _trial_population_spike_count  # noqa: E402

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_region_resolved_rate_stability_behaviour"

CTG_STEP = 3
CTG_N_SPLITS = 5
N_BOOT = 2000
N_PERM = 10000
N_SUBSAMPLE_DRAWS = 8
FDR_ALPHA = 0.05

# Region admission floor: reuses this project's own established per-session
# region floor (spike_pipeline.MIN_UNITS_PER_REGION == 8), applied here as
# the admission floor for a region cell too, so this analysis introduces no
# second, unexplained threshold. Every region in the table below clears this
# floor at the corpus level by a wide margin (the smallest, vmpfc/469, is
# 61 units); what the floor actually removes is individual UNDERSIZED
# SESSIONS within an otherwise-admitted region (tracked per cell below as
# n_sessions_refused_region_floor, kept separate from whole-session
# exclusions in n_sessions_refused_whole_session_excluded), and one whole
# region: dandi_000574's
# "unspecific" structure (466 units, 9 patients) is EXCLUDED even though it
# clears the unit floor by a wide margin, because it is not an anatomical
# assignment -- it is this corpus's own label for ambiguous/unlabelled
# microwire sites (src/spike_pipeline.py's BORAN_ANATOMICAL_REGIONS comment),
# and admitting it as a "region" would misrepresent what was measured.
RUTISHAUSER_REGIONS_469_673 = ANATOMICAL_REGIONS
RUTISHAUSER_REGIONS_001187 = ("hippocampus", "amygdala")
BORAN_REGIONS = tuple(r for r in BORAN_ANATOMICAL_REGIONS if r != "unspecific")
UNSPECIFIC_EXCLUSION_REASON = (
    "dandi_000574's 'unspecific' structure (466 units, 9 patients per "
    "results/anatomical_census.json) is excluded from region admission: it "
    "is this corpus's own label for ambiguous or unlabelled microwire "
    "sites, not an anatomical structure, so treating it as a region would "
    "misrepresent what was measured. It clears MIN_UNITS_PER_REGION by a "
    "wide margin and is excluded on anatomical grounds, not a power floor."
)


def _load_000469_session(path: Path) -> dict | None:
    with h5py.File(str(path), "r") as f:
        if "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        t_maint = trials["timestamps_Maintenance"][:]
        accuracy = trials["response_accuracy"][:].astype(bool)
    if len(loads) < MIN_TRIALS or accuracy.mean() < MIN_SESSION_ACCURACY:
        return None
    return dict(patient=path.parent.name, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                t_onset=t_maint, maint_win=2.3, condition=loads, low=1, high=3, accuracy=accuracy)


def _load_001187_session(path: Path) -> dict | None:
    with h5py.File(str(path), "r") as f:
        if "intervals" not in f or "WM_trials" not in f["intervals"] or "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/WM_trials"]
        loads = trials["loads"][:].astype(int)
        t_maint = trials["timestamps_Maintenance"][:]
        accuracy = trials["response_accuracy"][:].astype(bool)
    if len(loads) < MIN_TRIALS or accuracy.mean() < MIN_SESSION_ACCURACY:
        return None
    return dict(patient=path.parent.name, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                t_onset=t_maint, maint_win=2.3, condition=loads, low=1, high=3, accuracy=accuracy)


def _load_000673_session(path: Path) -> dict | None:
    with h5py.File(str(path), "r") as f:
        if "intervals" not in f or "trials" not in f["intervals"] or "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f)["region"]
        trials = f["intervals/trials"]
        loads = trials["loads"][:].astype(int)
        t_maint = trials["timestamps_Maintenance"][:]
        accuracy = trials["response_accuracy"][:].astype(bool)
    if len(loads) < MIN_TRIALS or accuracy.mean() < MIN_SESSION_ACCURACY:
        return None
    return dict(patient=path.parent.name, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                t_onset=t_maint, maint_win=2.3, condition=loads, low=1, high=3, accuracy=accuracy)


def _load_000574_session(path: Path) -> dict | None:
    with h5py.File(str(path), "r") as f:
        if "units" not in f:
            return None
        spike_lists_all = load_spike_times(f)
        unit_regions = resolve_unit_regions(f, "nwb_boran_brainnetome_hybrid")["region"]
        trials = f["intervals/trials"]
        set_size = trials["set_size"][:].astype(int)
        correct = trials["correct"][:].astype(bool)
        artifact = trials["artifact"][:].astype(bool)
        start_time = trials["start_time"][:]
    keep = ~artifact
    if keep.sum() < MIN_TRIALS:
        return None
    t_onset = start_time[keep] + 3.0
    return dict(patient=path.parent.name, spike_lists_all=spike_lists_all, unit_regions=unit_regions,
                t_onset=t_onset, maint_win=3.0, condition=set_size[keep], low=4, high=8,
                accuracy=correct[keep])


CORPUS_SPECS = {
    "dandi_000469": {
        "regions": RUTISHAUSER_REGIONS_469_673,
        "loader": _load_000469_session,
        "glob": lambda: sorted(dataset_path("dandi_000469").glob("sub-*/*_ses-2_ecephys+image.nwb")),
        "matched_pairs": [("hippocampus", "amygdala"), ("pre_sma", "dacc"), ("pre_sma", "vmpfc"), ("dacc", "vmpfc")],
    },
    "dandi_001187": {
        "regions": RUTISHAUSER_REGIONS_001187,
        "loader": _load_001187_session,
        "glob": lambda: sorted(dataset_path("dandi_001187").glob("sub-*/*_ecephys*.nwb")),
        "matched_pairs": [("hippocampus", "amygdala")],
    },
    "dandi_000673": {
        "regions": RUTISHAUSER_REGIONS_469_673,
        "loader": _load_000673_session,
        "glob": lambda: sorted(dataset_path("dandi_000673").glob("sub-*/*_ecephys*.nwb")),
        "matched_pairs": [("hippocampus", "amygdala"), ("pre_sma", "dacc"), ("pre_sma", "vmpfc"), ("dacc", "vmpfc")],
    },
    "dandi_000574": {
        "regions": BORAN_REGIONS,
        "loader": _load_000574_session,
        "glob": lambda: sorted(dataset_path("dandi_000574").glob("sub-0[1-9]/*.nwb")),
        "matched_pairs": [("hippocampus", "amygdala")],
    },
}




def _fit_ctg_offdiag(psth_z: np.ndarray, condition: np.ndarray, low: int, high: int,
                      n_components: int, rng: np.random.Generator) -> dict | None:
    """Off-diagonal cross-temporal-generalisation effect, no permutation null.

    This artifact's significance tests all run at the patient level
    (pearson/partial-correlation permutation tests, paired sign-flip tests) --
    a per-session label-permutation null on top of that would be a redundant
    second null answering a question (is THIS session's decoding above
    chance) this artifact never asks. ctg_nested_cv/temporal_stability_tau
    are the same estimator core scripts/run_ctg_offdiagonal_temporal_stability.py
    reads (via the pooled per-corpus *_ctg_*.npz files); this only skips the
    permutation-null wrapper those pooled artifacts also compute.
    """
    mask = (condition == low) | (condition == high)
    if min((condition[mask] == low).sum(), (condition[mask] == high).sum()) < 5:
        return None
    X = psth_z[mask]
    y = (condition[mask] == high).astype(int)
    t_idx = np.arange(0, psth_z.shape[2], CTG_STEP)
    if len(t_idx) < 2:
        return None
    auc_mat = ctg_nested_cv(X, y, t_idx, n_components=n_components, n_splits=CTG_N_SPLITS, rng=rng)
    return temporal_stability_tau(auc_mat)


def _region_cell(session: dict, region: str) -> dict:
    spike_lists = region_filtered_units(session["spike_lists_all"], session["unit_regions"], region,
                                         session["t_onset"], session["maint_win"])
    if spike_lists is None:
        return {"status": "refused", "reason": f"fewer than {MIN_UNITS_PER_REGION} units in "
                f"{region} surviving {MIN_UNIT_FIRING_RATE_HZ} Hz rate QC"}
    n_units = len(spike_lists)
    rates = unit_mean_firing_rates(spike_lists, session["t_onset"], session["maint_win"])
    psth = build_psth(spike_lists, session["t_onset"], bin_ms=BIN_MS_DEFAULT, smooth_ms=SMOOTH_MS_DEFAULT,
                       window_s=session["maint_win"])
    psth_z = FrozenPSTHTransform().fit_transform(psth)
    n_comp = max(2, min(N_PC_DEFAULT, n_units - 1))
    rng = np.random.default_rng(stable_seed(f"{session['patient']}_{region}"))
    tau_info = _fit_ctg_offdiag(psth_z, session["condition"], session["low"], session["high"], n_comp, rng)
    spike_count = _trial_population_spike_count(spike_lists, session["t_onset"], session["maint_win"])
    accuracy = np.asarray(session["accuracy"], dtype=bool)
    return {
        "status": "computed",
        "n_units": n_units,
        "n_trials": int(len(accuracy)),
        "n_correct": int(accuracy.sum()),
        "median_unit_rate_hz": float(np.median(rates)),
        "mean_trial_spike_count": float(np.mean(spike_count)),
        "ctg_status": "computed" if tau_info is not None else "underpowered_fewer_than_5_trials_per_condition",
        "ctg": None if tau_info is None else {
            "offdiag_effect": tau_info["offdiag_effect"],
            "mean_diag_auc": tau_info["mean_diag_auc"],
            "tau": tau_info["tau"],
            "interpretable": tau_info["interpretable"],
        },
    }


def _matched_pair_session(session: dict, region_a: str, region_b: str, rng: np.random.Generator) -> dict:
    units_a = filter_units_by_region(session["spike_lists_all"], session["unit_regions"], region_a)
    units_b = filter_units_by_region(session["spike_lists_all"], session["unit_regions"], region_b)
    mask_a = low_rate_unit_mask(units_a, session["t_onset"], session["maint_win"])
    mask_b = low_rate_unit_mask(units_b, session["t_onset"], session["maint_win"])
    units_a = [u for u, k in zip(units_a, mask_a) if k]
    units_b = [u for u, k in zip(units_b, mask_b) if k]
    target = min(len(units_a), len(units_b))
    if target < MIN_UNITS_PER_REGION:
        return {"status": "refused", "reason": f"matched target {target} units below floor "
                f"{MIN_UNITS_PER_REGION} ({region_a}={len(units_a)}, {region_b}={len(units_b)})"}
    draws = {"rate_a": [], "rate_b": [], "stability_a": [], "stability_b": []}
    n_comp = max(2, min(N_PC_DEFAULT, target - 1))
    for _ in range(N_SUBSAMPLE_DRAWS):
        idx_a = rng.choice(len(units_a), target, replace=False)
        idx_b = rng.choice(len(units_b), target, replace=False)
        sub_a = [units_a[i] for i in idx_a]
        sub_b = [units_b[i] for i in idx_b]
        draws["rate_a"].append(float(np.median(unit_mean_firing_rates(sub_a, session["t_onset"], session["maint_win"]))))
        draws["rate_b"].append(float(np.median(unit_mean_firing_rates(sub_b, session["t_onset"], session["maint_win"]))))
        psth_a = build_psth(sub_a, session["t_onset"], bin_ms=BIN_MS_DEFAULT, smooth_ms=SMOOTH_MS_DEFAULT, window_s=session["maint_win"])
        psth_b = build_psth(sub_b, session["t_onset"], bin_ms=BIN_MS_DEFAULT, smooth_ms=SMOOTH_MS_DEFAULT, window_s=session["maint_win"])
        z_a = FrozenPSTHTransform().fit_transform(psth_a)
        z_b = FrozenPSTHTransform().fit_transform(psth_b)
        info_a = _fit_ctg_offdiag(z_a, session["condition"], session["low"], session["high"], n_comp, rng)
        info_b = _fit_ctg_offdiag(z_b, session["condition"], session["low"], session["high"], n_comp, rng)
        draws["stability_a"].append(info_a["offdiag_effect"] if info_a is not None else None)
        draws["stability_b"].append(info_b["offdiag_effect"] if info_b is not None else None)
    return {
        "status": "computed", "target_n_units": int(target),
        "rate_a": pool_draws_within_session(draws["rate_a"]), "rate_b": pool_draws_within_session(draws["rate_b"]),
        "stability_a": pool_draws_within_session(draws["stability_a"]),
        "stability_b": pool_draws_within_session(draws["stability_b"]),
    }


def _checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.json"


def _load_checkpoint(name: str) -> dict:
    path = _checkpoint_path(name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text()).get("sessions", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _record_session(name: str, session_key: str, record: dict) -> None:
    with locked_json_update(_checkpoint_path(name)) as ck:
        ck.setdefault("sessions", {})[session_key] = record


def process_corpus(corpus: str) -> None:
    spec = CORPUS_SPECS[corpus]
    region_done = {r: _load_checkpoint(f"{corpus}__{r}") for r in spec["regions"]}
    pair_done = {(a, b): _load_checkpoint(f"{corpus}__matched__{a}_vs_{b}") for a, b in spec["matched_pairs"]}
    for path in spec["glob"]():
        session_key = path.stem
        need_region = [r for r in spec["regions"] if session_key not in region_done[r]]
        need_pair = [(a, b) for (a, b) in spec["matched_pairs"] if session_key not in pair_done[(a, b)]]
        if not need_region and not need_pair:
            continue
        session = spec["loader"](path)
        if session is None:
            reason = {"status": "refused", "reason": "session excluded: below MIN_TRIALS or "
                      "MIN_SESSION_ACCURACY, or missing units/trials tables"}
            for r in need_region:
                _record_session(f"{corpus}__{r}", session_key, reason)
                region_done[r][session_key] = reason
            for a, b in need_pair:
                _record_session(f"{corpus}__matched__{a}_vs_{b}", session_key, reason)
                pair_done[(a, b)][session_key] = reason
            continue
        for r in need_region:
            cell = _region_cell(session, r)
            cell["patient"] = session["patient"]
            _record_session(f"{corpus}__{r}", session_key, cell)
            region_done[r][session_key] = cell
        for a, b in need_pair:
            rng = np.random.default_rng(stable_seed(f"{session_key}_{a}_vs_{b}"))
            cell = _matched_pair_session(session, a, b, rng)
            cell["patient"] = session["patient"]
            _record_session(f"{corpus}__matched__{a}_vs_{b}", session_key, cell)
            pair_done[(a, b)][session_key] = cell
        print(f"  {corpus} {session_key}: regions done", flush=True)


def _patient_rows(region_sessions: dict) -> list:
    by_patient: dict[str, list] = {}
    for cell in region_sessions.values():
        if cell.get("status") != "computed":
            continue
        by_patient.setdefault(cell["patient"], []).append(cell)
    rows = []
    for patient, cells in sorted(by_patient.items()):
        stab = [c["ctg"]["offdiag_effect"] for c in cells if c.get("ctg") is not None]
        n_trials = sum(c["n_trials"] for c in cells)
        n_correct = sum(c["n_correct"] for c in cells)
        rows.append({
            "patient": patient, "n_sessions": len(cells),
            "n_sessions_with_stability": len(stab),
            "n_units_median": float(np.median([c["n_units"] for c in cells])),
            "rate_hz": float(np.mean([c["median_unit_rate_hz"] for c in cells])),
            "stability_offdiag_effect": float(np.mean(stab)) if stab else None,
            "accuracy": float(n_correct) / n_trials if n_trials else None,
            "n_trials": n_trials,
            "spike_count": float(np.mean([c["mean_trial_spike_count"] for c in cells])),
        })
    return rows


def _partial_r_stat(y: np.ndarray, x: np.ndarray, controls: list) -> float:
    n = len(y)
    if controls:
        design = np.column_stack([np.ones(n), *controls])
        y_r = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
        x_r = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    else:
        y_r, x_r = y - y.mean(), x - x.mean()
    if np.std(y_r) == 0.0 or np.std(x_r) == 0.0:
        return float("nan")
    return float(np.corrcoef(y_r, x_r)[0, 1])


def _correlation_cell(x: list, y: list, controls: list, seed_tag: str) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    control_arrs = [np.asarray(c, dtype=float) for c in controls]
    n = len(x)
    if n < 4:
        return {"status": "underpowered", "n": n,
                "reason": "fewer than 4 patients carry both quantities computed"}
    rng = np.random.default_rng(stable_seed(seed_tag))

    def _stat(idx_block: np.ndarray) -> float:
        idx = idx_block[:, 0].astype(int)
        return _partial_r_stat(y[idx], x[idx], [c[idx] for c in control_arrs])

    data = np.arange(n).reshape(-1, 1)
    r_obs, lo, hi = bootstrap_ci(data, _stat, n_boot=N_BOOT, rng=rng)
    if not np.isfinite(r_obs):
        return {"status": "not_computable", "n": n, "reason": "zero-variance residual after controls"}
    se = (hi - lo) / (2 * 1.959963984540054)
    mdd = float(Z_80_POWER * se) if np.isfinite(se) else float("nan")
    if control_arrs:
        test = partial_correlation_permutation_test(y, x, controls=control_arrs, n_perm=N_PERM, rng=rng)
        p_value = test.get("p_value", float("nan"))
    else:
        test = pearson_permutation_test(x, y, n_perm=N_PERM, rng=rng)
        p_value = test["p_value"]
    return {"status": "computed", "n": n, "estimate": r_obs, "ci_lower": lo, "ci_upper": hi,
            "mdd": mdd, "p_value": p_value, "n_controls": len(control_arrs)}


WITHIN_REGION_FAMILIES = ("rate_to_stability", "rate_to_behaviour", "stability_to_behaviour")


def _within_region_cells(rows: list, corpus: str, region: str) -> dict:
    cells = {}
    for family, xk, yk, need_stab in (
        ("rate_to_stability", "rate_hz", "stability_offdiag_effect", True),
        ("rate_to_behaviour", "rate_hz", "accuracy", False),
        ("stability_to_behaviour", "stability_offdiag_effect", "accuracy", True),
    ):
        usable = [r for r in rows if r[xk] is not None and r[yk] is not None]
        x = [r[xk] for r in usable]
        y = [r[yk] for r in usable]
        sc = [r["spike_count"] for r in usable]
        tag = f"{corpus}|{region}|{family}"
        raw = _correlation_cell(x, y, [], f"{tag}|raw")
        controlled = _correlation_cell(x, y, [sc], f"{tag}|spike_count_controlled")
        cells[family] = {"raw": raw, "spike_count_controlled": controlled,
                          "n_patients_region_total": len(rows), "n_patients_used": len(usable)}
    return cells


def _matched_pair_patient_rows(pair_sessions: dict) -> list:
    by_patient: dict[str, list] = {}
    for cell in pair_sessions.values():
        if cell.get("status") != "computed":
            continue
        by_patient.setdefault(cell["patient"], []).append(cell)
    rows = []
    for patient, cells in sorted(by_patient.items()):
        row = {"patient": patient, "n_sessions": len(cells)}
        for key in ("rate_a", "rate_b", "stability_a", "stability_b"):
            vals = [c[key] for c in cells if c.get(key) is not None]
            row[key] = float(np.mean(vals)) if vals else None
        rows.append(row)
    return rows


def _paired_contrast(rows: list, key_a: str, key_b: str) -> dict:
    diffs = [r[key_a] - r[key_b] for r in rows if r[key_a] is not None and r[key_b] is not None]
    if len(diffs) < 4:
        return {"status": "underpowered", "n_patients": len(diffs),
                "reason": "fewer than 4 patients carry both regions' matched estimate"}
    test = slope_across_sessions_test(diffs)
    mdd = minimum_detectable_paired_difference(diffs)
    return {"status": test["status"], "n_patients": len(diffs),
            "mean_difference_a_minus_b": test.get("mean_value"), "p_value": test.get("p_value"),
            "ci_lower": test.get("ci_lower"), "ci_upper": test.get("ci_upper"),
            "mdd": mdd.get("mdd"), "values_a": [r[key_a] for r in rows if r[key_a] is not None],
            "values_b": [r[key_b] for r in rows if r[key_b] is not None]}


def _region_estimate_summary(values: list) -> dict:
    values = [v for v in values if v is not None and np.isfinite(v)]
    if len(values) < 2:
        return {"status": "not_computable", "n": len(values)}
    return {"status": "computed", "n": len(values), "median": float(np.median(values)),
            "min": float(np.min(values)), "max": float(np.max(values))}


def build_artifact() -> dict:
    predeclared_rules = {
        "region_admission_floor": f"MIN_UNITS_PER_REGION={MIN_UNITS_PER_REGION} (src/spike_pipeline.py, "
            "this project's own existing per-session region floor, reused unmodified here as the "
            "admission floor for a region cell); dandi_000574's 'unspecific' structure is separately "
            "excluded on anatomical grounds regardless of unit count -- see unspecific_exclusion_reason.",
        "stability_estimator": "cross-temporal-generalisation off-diagonal effect "
            "(offdiag(AUC-0.5), src/geometry.py:ctg_nested_cv + temporal_stability_tau), the same "
            "instrument results/ctg_offdiagonal_temporal_stability.json reads from this project's "
            "pooled per-corpus fits, fit here per (session, region) instead of pooled across the "
            "whole recording. No session-level label-permutation null is refit here (see "
            "_fit_ctg_offdiag docstring) -- every significance test in this artifact runs at the "
            "patient level.",
        "resampling_unit": "patient; sessions of one patient are averaged into one value per patient "
            "before every correlation, bootstrap, and paired test below -- never the unit, never the trial.",
        "spike_count_control": "partial_correlation_permutation_test (src/statistics.py) controlling "
            "for per-patient mean per-trial population spike count, reported alongside the raw "
            "estimate for every within-region family, matching this project's existing spike-count "
            "control convention. For the two rate-linked families (rate_to_stability, "
            "rate_to_behaviour) this control is close to collinear with the family's own covariate "
            "(rate is spike count normalised by duration and unit count), so a near-zero controlled "
            "estimate there is an expected consequence of that near-collinearity, not evidence the "
            "raw coupling was spurious -- disclosed per cell, not silently applied.",
        "unit_count_control": "between-region contrasts are fit ONLY at a common, per-session "
            "subsampled unit count (target = min available rate-QC'd units across the two regions "
            "being compared that session), never on the raw unmatched counts; "
            f"{N_SUBSAMPLE_DRAWS} draws per session, pooled within session via "
            "run_count_subsampling_ladder.pool_draws_within_session before any cross-patient test.",
        "unspecific_exclusion_reason": UNSPECIFIC_EXCLUSION_REASON,
        "multiplicity_families": {
            "within_region_rate_to_stability": "one cell per (corpus, region)",
            "within_region_rate_to_behaviour": "one cell per (corpus, region)",
            "within_region_stability_to_behaviour": "one cell per (corpus, region)",
            "between_region_count_matched": "one cell per (corpus, region_a, region_b) pair",
        },
        "declared_before_fitting": True,
    }
    artifact = {
        "description": "Region-resolved firing rate, within-maintenance-window state stability, and "
            "trial-accuracy behaviour in four human microwire single-unit epilepsy corpora, fit "
            "separately per anatomical region.",
        "code_commit": git_commit(ROOT),
        "predeclared_rules": predeclared_rules,
        "corpora": {},
        "status": "in_progress",
    }
    _write(artifact)

    for corpus in CORPUS_SPECS:
        print(f"processing {corpus}", flush=True)
        process_corpus(corpus)

    all_cells_for_fdr = {"within_region_rate_to_stability": [], "within_region_rate_to_behaviour": [],
                          "within_region_stability_to_behaviour": [], "between_region_count_matched": []}

    for corpus, spec in CORPUS_SPECS.items():
        corpus_entry = {"regions": {}, "matched_pairs": {}, "independent_group": resolve_group(corpus)}
        for region in spec["regions"]:
            region_sessions = _load_checkpoint(f"{corpus}__{region}")
            rows = _patient_rows(region_sessions)
            n_refused_whole_session = sum(1 for c in region_sessions.values()
                                           if c.get("status") == "refused" and "MIN_TRIALS" in c.get("reason", ""))
            n_refused_region_floor = sum(1 for c in region_sessions.values()
                                          if c.get("status") == "refused" and "MIN_TRIALS" not in c.get("reason", ""))
            within_cells = _within_region_cells(rows, corpus, region)
            for family, cell in within_cells.items():
                key = f"within_region_{family}"
                if cell["raw"].get("status") == "computed":
                    all_cells_for_fdr[key].append(cell["raw"])
                if cell["spike_count_controlled"].get("status") == "computed":
                    all_cells_for_fdr[key].append(cell["spike_count_controlled"])
            corpus_entry["regions"][region] = {
                "n_sessions_total": len(region_sessions),
                "n_sessions_computed": sum(1 for c in region_sessions.values() if c.get("status") == "computed"),
                "n_sessions_refused": n_refused_whole_session + n_refused_region_floor,
                "n_sessions_refused_whole_session_excluded": n_refused_whole_session,
                "n_sessions_refused_region_floor": n_refused_region_floor,
                "n_patients": len(rows),
                "n_units_per_patient": _region_estimate_summary([r["n_units_median"] for r in rows]),
                "firing_rate_hz": _region_estimate_summary([r["rate_hz"] for r in rows]),
                "stability_offdiag_effect": _region_estimate_summary([r["stability_offdiag_effect"] for r in rows]),
                "accuracy": _region_estimate_summary([r["accuracy"] for r in rows]),
                "families": within_cells,
            }
        for region_a, region_b in spec["matched_pairs"]:
            pair_sessions = _load_checkpoint(f"{corpus}__matched__{region_a}_vs_{region_b}")
            rows = _matched_pair_patient_rows(pair_sessions)
            rate_contrast = _paired_contrast(rows, "rate_a", "rate_b")
            stab_contrast = _paired_contrast(rows, "stability_a", "stability_b")
            for contrast in (rate_contrast, stab_contrast):
                if contrast.get("status") == "tested":
                    all_cells_for_fdr["between_region_count_matched"].append(contrast)
            corpus_entry["matched_pairs"][f"{region_a}_vs_{region_b}"] = {
                "n_sessions_total": len(pair_sessions),
                "n_sessions_computed": sum(1 for c in pair_sessions.values() if c.get("status") == "computed"),
                "n_patients_matched": len(rows),
                "firing_rate_matched_count": rate_contrast,
                "stability_matched_count": stab_contrast,
                "region_a": region_a, "region_b": region_b,
            }
        artifact["corpora"][corpus] = corpus_entry
        _write(artifact)

    for family, cells in all_cells_for_fdr.items():
        if not cells:
            continue
        q = fdr_bh(np.array([c["p_value"] for c in cells]), alpha=FDR_ALPHA)
        for cell, qv in zip(cells, q["q_values"]):
            cell["q_value_fdr"] = float(qv)

    artifact["independence"] = {
        "n_independent_groups": count_independent_groups(list(CORPUS_SPECS)),
        "groups": {corpus: resolve_group(corpus) for corpus in CORPUS_SPECS},
        "note": "dandi_000673 and dandi_001187 resolve to one independence group; dandi_000469 and "
            "dandi_000574 are each their own group -- three independence groups across four corpus "
            "cells. Any statement about how many independent sources support a regional pattern must "
            "count these three groups, not the four corpus cells.",
        "hippocampus_amygdala_coverage": "all four corpus cells (three independence groups) carry a "
            "hippocampus_vs_amygdala matched-pair cell -- the contrast this artifact reports most "
            "carefully.",
        "frontal_region_coverage": "pre_sma/dacc/vmpfc matched pairs exist only in dandi_000469 and "
            "dandi_000673, which are two independence groups (000469 alone, and the "
            "000673/001187 group via 000673) -- one-to-two independence groups, not a broad "
            "replication.",
        "behaviour_link_baseline_note": "scripts/run_human_maintenance_behaviour_link.py's existing "
            "52-patient pooled behaviour-link figure is computed over dandi_000469, dandi_001187, "
            "and dandi_000574 only -- dandi_000673 was never part of it. This artifact's cells that "
            "include dandi_000673 extend that baseline rather than reproducing it.",
    }
    artifact["status"] = "complete"
    _write(artifact)
    return artifact


def _write(artifact: dict) -> None:
    with open(RESULTS / "region_resolved_rate_stability_behaviour.json", "w") as f:
        json.dump(_json_safe(artifact), f, indent=2, allow_nan=False)


def main():
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    build_artifact()


if __name__ == "__main__":
    main()
