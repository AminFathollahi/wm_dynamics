"""run_instrument_matched_content_decodability.py -- is the human memorandum-decodability
null (results/human_content_decodability.json) an instrument artifact of unit count, or
does it survive matching the animal arms to the human arms' unit and trial counts?

The mouse (inagaki_alm5) and macaque (panichello_2024) content-decodability arms are
already matched to the human dandi_001187 arm on trial count (136 vs 137) and on the
decoder's rank parameter (k = 8 in every session of both). The one remaining instrument
variable is unit count: 23 in the human arm versus 41 (mouse) and 317 (macaque). This
module subsamples each animal session's units -- and, in a second and third target, its
trials as well -- down to a draw from the realised empirical distribution of a human arm,
recomputes the rank parameter k after the cut (never carries it over), and reruns the
identical, unmodified decodability estimator (session_subtractive_test, imported from
scripts/run_state_content_link.py) on the cut population. If the animal arms still decode
the memorandum after the cut, the human null is a property of the preparation and not of
the instrument. If they do not, the cross-preparation contrast is uninterpretable and is
reported as such rather than stated as a species or preparation claim.

Three matched targets, each drawn from a human arm's REALISED distribution (not its
median alone) so the matched arm reproduces the human spread:
  dandi_001187_units_only        -- unit count only; donor keeps its own trial count.
  dandi_001187_units_and_trials  -- unit count and trial count both matched.
  dandi_000469_units_and_trials  -- unit count and trial count both matched to the
                                     other human arm, the one with an unmatched k.
Two donors (mouse, macaque) times three targets gives six cells. Each (donor session,
cell) pair draws its target unit/trial counts 20 times with replacement from the human
arm's empirical list, subsamples without replacement (trials stratified by class to hold
the donor's own class proportions), and reruns the estimator; the per-session value
reported is the median across those 20 draws, with the 5th/95th percentile as the spread,
per session -- not one draw. A draw whose target exceeds what the donor session has
available is unreachable and is recorded, not silently dropped.

Decision rules, pre-declared before any output of this module was inspected:

  decodability_survives_instrument_matching -- the cut arm's median a_full 95% bootstrap
    interval (over per-session medians) excludes the target human arm's own median a_full,
    AND the exact (Clopper-Pearson) interval on the fraction of donor sessions clearing
    their own permutation null is above one half.
  decodability_lost_to_instrument_matching -- the cut arm's median a_full interval covers
    the target human arm's median a_full.
  inconclusive_below_detection_floor -- fewer than 8 donor sessions produced at least one
    fitted draw after the cut, OR (pre-declared fallback for the one case the two rules
    above do not jointly resolve) the median-a_full criterion excludes the human value but
    the clearing-fraction criterion does not clear one half -- the two criteria disagree
    and neither named branch is fully satisfied, so no positive claim is reported.

Reachability is fixed before any data is touched, not discovered after the fact: the
full-model permutation null inside session_subtractive_test always draws
CONTENT_N_PERM_FULL=50 replicates regardless of the subsampled trial count, so the
smallest attainable per-session permutation p is 1/51 = 0.0196, below the 0.05 threshold,
for every draw that fits at all (k >= 3). The permutation floor is therefore always
reachable here; the binding reachability constraint in this design is the session-count
floor above, not the permutation replicate count.
"""

from __future__ import annotations

import os

for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import glob  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.io import loadmat  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from corpus_sessions import data_root, iter_alm  # noqa: E402
from provenance import _json_safe  # noqa: E402
from run_human_content_decodability import MIN_SESSIONS_FOR_PRIMARY_BRANCH  # noqa: E402
from run_state_content_link import (  # noqa: E402
    BIN_MS,
    CONTENT_N_PERM_FULL,
    MIN_CLASSES,
    MIN_TRIALS_PER_CLASS,
    PANICHELLO_DELAY_WINDOW_MS,
    _panichello_directory,
    _stable_seed,
    session_subtractive_test,
    usable_label,
)
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from statistics import bootstrap_ci  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "results" / "instrument_matched_content_decodability.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / ".checkpoints" / "instrument_matched_content_decodability_checkpoint.json"
HUMAN_DECODABILITY_PATH = REPO_ROOT / "results" / "human_content_decodability.json"
CONTENT_LINK_PATH = REPO_ROOT / "results" / "state_content_link.json"
CARDINALITY_LADDER_PATH = REPO_ROOT / "results" / "content_label_cardinality_ladder.json"

ALPHA = 0.05
CHANCE_LEVEL = 0.5
BOOTSTRAP_SEED = 20260814
N_BOOT = 2000
N_DRAWS_PER_SESSION_PER_CELL = 20
MAX_WORKERS = 8

DECLARED_BRANCHES = (
    "decodability_survives_instrument_matching",
    "decodability_lost_to_instrument_matching",
    "inconclusive_below_detection_floor",
)
DECISION_RULE = (
    "Reachability first: with fewer than %d donor sessions producing at least one fitted "
    "draw after the cut, inconclusive_below_detection_floor. Otherwise: "
    "decodability_survives_instrument_matching if the cut arm's bootstrap 95%% interval on "
    "the median a_full (over per-session medians) excludes the target human arm's own "
    "median a_full AND the exact interval on the fraction of donor sessions clearing their "
    "own permutation null is above one half; decodability_lost_to_instrument_matching if "
    "the median a_full interval covers the target human arm's median; otherwise (the two "
    "criteria disagree) inconclusive_below_detection_floor." % MIN_SESSIONS_FOR_PRIMARY_BRANCH
)

MIN_ATTAINABLE_PERMUTATION_P = 1.0 / (CONTENT_N_PERM_FULL + 1)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, default=_json_default))
    tmp.replace(path)


# ── Donor session loading (raw per-trial, per-unit spike counts) ──────────────

def load_mouse_sessions(root: Path) -> list[dict]:
    out = []
    for meta in iter_alm(root, bin_ms=BIN_MS, window_s=2.0):
        out.append({
            "donor": "inagaki_alm5", "session": meta["session"],
            "counts": np.asarray(meta["counts"], dtype=float),
            "labels": np.asarray(meta["condition"]),
        })
    return out


def load_macaque_sessions(root: Path) -> list[dict]:
    """Reproduces the panichello_2024 ingestion in run_state_content_link.py
    (correct trials only, 100 ms bins over the 300-1450 ms delay window) so
    this module's donor arm is measured on the identical raw population."""
    out = []
    directory = _panichello_directory(root)
    if directory is None:
        return out
    for path in sorted(glob.glob(str(directory / "*.mat"))):
        raw = loadmat(path, squeeze_me=True)
        spikes = np.asarray(raw["spks"], dtype=float)
        time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
        correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
        cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
        spikes, cue_idx = spikes[correct], cue_idx[correct]
        starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], BIN_MS)
        binned = [spikes[:, (time_ms >= s) & (time_ms < s + BIN_MS), :].sum(axis=1) for s in starts]
        counts = np.stack(binned, axis=2)
        out.append({
            "donor": "panichello_2024", "session": Path(path).stem,
            "counts": counts, "labels": cue_idx.astype(int),
        })
    return out


# ── Human target distributions, read from the delivered artifacts ─────────────

def load_targets() -> dict:
    human = json.loads(HUMAN_DECODABILITY_PATH.read_text())
    content_link = json.loads(CONTENT_LINK_PATH.read_text())

    rows_1187 = human["session_rows_dandi_001187"]
    units_1187 = [int(r["n_units"]) for r in rows_1187]
    trials_1187 = [int(r["n_trials"]) for r in rows_1187]
    arm_1187 = human["primary_decodability"]["per_arm"]["dandi_001187"]

    rows_469 = [r for r in content_link["session_rows"]
                if r["dataset"] == "dandi_000469" and r["structure"] == "pooled"]
    units_469 = [int(r["n_units"]) for r in rows_469]
    trials_469 = [int(r["n_trials"]) for r in rows_469]
    arm_469 = human["primary_decodability"]["per_arm"]["dandi_000469"]

    return {
        "dandi_001187_units_only": {
            "unit_pool": units_1187, "trial_pool": None,
            "human_dataset": "dandi_001187", "n_human_sessions": len(rows_1187),
            "human_median_a_full": arm_1187["median_a_full"]["median"],
            "human_median_a_full_ci95": arm_1187["median_a_full"]["ci95"],
        },
        "dandi_001187_units_and_trials": {
            "unit_pool": units_1187, "trial_pool": trials_1187,
            "human_dataset": "dandi_001187", "n_human_sessions": len(rows_1187),
            "human_median_a_full": arm_1187["median_a_full"]["median"],
            "human_median_a_full_ci95": arm_1187["median_a_full"]["ci95"],
        },
        "dandi_000469_units_and_trials": {
            "unit_pool": units_469, "trial_pool": trials_469,
            "human_dataset": "dandi_000469", "n_human_sessions": len(rows_469),
            "human_median_a_full": arm_469["median_a_full"]["median"],
            "human_median_a_full_ci95": arm_469["median_a_full"]["ci95"],
        },
    }


# ── One subsampled draw ────────────────────────────────────────────────────────

def _stratified_trial_indices(labels: np.ndarray, n_target: int, rng: np.random.Generator) -> np.ndarray:
    """Subsample n_target trial indices without replacement, holding the donor
    session's own class proportions (largest-remainder allocation, then
    per-class sampling without replacement)."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    raw = counts.astype(float) * n_target / total
    base = np.floor(raw).astype(int)
    deficit = int(n_target - base.sum())
    if deficit > 0:
        order = np.argsort(-(raw - base))
        for i in order[:deficit]:
            base[i] += 1
    picked = []
    for cls, want in zip(classes, base):
        if want <= 0:
            continue
        cls_idx = np.flatnonzero(labels == cls)
        picked.append(rng.choice(cls_idx, size=int(want), replace=False))
    return np.sort(np.concatenate(picked)) if picked else np.array([], dtype=int)


def _one_draw(counts: np.ndarray, labels: np.ndarray, unit_pool: list[int],
              trial_pool: list[int] | None, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_trials_avail, n_units_avail = counts.shape[0], counts.shape[1]

    unit_target = int(rng.choice(unit_pool))
    if unit_target > n_units_avail:
        return {"status": "unit_target_exceeds_available_units",
                "unit_target": unit_target, "n_units_available": n_units_avail}
    unit_idx = np.sort(rng.choice(n_units_avail, size=unit_target, replace=False))

    if trial_pool is not None:
        trial_target = int(rng.choice(trial_pool))
        if trial_target > n_trials_avail:
            return {"status": "trial_target_exceeds_available_trials",
                    "trial_target": trial_target, "n_trials_available": n_trials_avail}
        trial_idx = _stratified_trial_indices(labels, trial_target, rng)
    else:
        trial_target = n_trials_avail
        trial_idx = np.arange(n_trials_avail)

    sub_labels = labels[trial_idx]
    ok, reason, mask = usable_label(sub_labels, min_classes=MIN_CLASSES, min_per_class=MIN_TRIALS_PER_CLASS)
    if not ok:
        return {"status": "no_usable_label_after_cut", "reason": reason,
                "unit_target": unit_target, "trial_target": trial_target}
    trial_idx, sub_labels = trial_idx[mask], sub_labels[mask]
    sub_counts = counts[trial_idx][:, unit_idx, :]

    window_mean = FrozenPSTHTransform().fit(sub_counts).transform(sub_counts).mean(axis=2)[:, :, None]
    fit = session_subtractive_test(window_mean, sub_labels, seed)
    classes, class_n = np.unique(sub_labels, return_counts=True)
    return {
        "status": fit.get("status", "unknown"),
        "unit_target": unit_target, "trial_target": trial_target,
        "n_units_realised": int(sub_counts.shape[1]), "n_trials_realised": int(sub_counts.shape[0]),
        "class_counts_realised": {int(c): int(n) for c, n in zip(classes, class_n)},
        "fit": fit,
    }


def _session_worker(donor: str, session: str, counts: np.ndarray, labels: np.ndarray,
                     targets: dict, existing_keys: set) -> dict:
    """All not-yet-checkpointed draws for one donor session, across every
    target cell. Run inside a worker process; the caller merges the result
    dict into the checkpoint and writes it once the whole session returns."""
    out = {}
    for target_name, cfg in targets.items():
        for draw in range(N_DRAWS_PER_SESSION_PER_CELL):
            key = f"{donor}|{session}|{target_name}|draw{draw}"
            if key in existing_keys:
                continue
            seed = _stable_seed(donor, session, target_name, draw)
            record = _one_draw(counts, labels, cfg["unit_pool"], cfg["trial_pool"], seed)
            record.update({"donor": donor, "session": session, "target": target_name,
                            "draw": draw, "seed": int(seed)})
            out[key] = record
    return out


# ── Checkpointing ──────────────────────────────────────────────────────────────

def _fingerprint(targets: dict) -> dict:
    return {
        "n_draws_per_session_per_cell": N_DRAWS_PER_SESSION_PER_CELL,
        "targets": {name: {"unit_pool": cfg["unit_pool"], "trial_pool": cfg["trial_pool"]}
                    for name, cfg in targets.items()},
    }


def load_checkpoint(targets: dict) -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        cached = json.loads(CHECKPOINT_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if cached.get("fingerprint") != _fingerprint(targets):
        return {}
    return cached.get("draws", {})


def save_checkpoint(targets: dict, draws: dict) -> None:
    _write_json(CHECKPOINT_PATH, {"fingerprint": _fingerprint(targets), "n_draws": len(draws), "draws": draws})


# ── Per-cell aggregation ────────────────────────────────────────────────────────

def _status_counts(values: list[str]) -> dict:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def per_session_summary(session: str, draws: list[dict]) -> dict:
    unreachable = {"unit_target_exceeds_available_units", "trial_target_exceeds_available_trials"}
    fitted = [d for d in draws if d["status"] == "tested"]
    n_reachable = sum(1 for d in draws if d["status"] not in unreachable)
    a_full_vals = [d["fit"]["a_full"] for d in fitted]
    p_vals = [d["fit"]["a_full_p_value"] for d in fitted]
    k_vals = [d["fit"]["k_latents"] for d in fitted]
    clearing_frac = float(np.mean([p <= ALPHA for p in p_vals])) if p_vals else None
    return {
        "session": session, "n_draws_attempted": len(draws), "n_draws_reachable": n_reachable,
        "n_draws_fitted": len(fitted), "status_counts": _status_counts([d["status"] for d in draws]),
        "a_full_median": float(np.median(a_full_vals)) if a_full_vals else None,
        "a_full_p5_p95": ([float(np.percentile(a_full_vals, 5)), float(np.percentile(a_full_vals, 95))]
                           if a_full_vals else None),
        "fraction_draws_clearing_own_null": clearing_frac,
        "session_clears_own_null_majority": (clearing_frac is not None and clearing_frac > 0.5),
        "k_distribution": _status_counts([str(k) for k in k_vals]),
        "fraction_draws_k_equals_8": (float(np.mean([k == 8 for k in k_vals])) if k_vals else None),
    }


def _median_with_ci(values: list[float], seed_offset: int) -> dict:
    if len(values) < 2:
        return {"status": "point_estimate_only", "n": len(values),
                "median": (float(values[0]) if values else None)}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    observed, low, high = bootstrap_ci(np.asarray(values, dtype=float), np.median, n_boot=N_BOOT, rng=rng)
    return {"status": "computed", "n": len(values), "median": observed, "ci95": [low, high], "n_boot": N_BOOT}


def cell_analysis(donor: str, target_name: str, target_cfg: dict, donor_sessions: list[dict],
                   draws_by_key: dict, seed_offset: int) -> dict:
    per_session = []
    all_fitted_k = []
    for s in donor_sessions:
        session = s["session"]
        rows = [draws_by_key[f"{donor}|{session}|{target_name}|draw{d}"]
                for d in range(N_DRAWS_PER_SESSION_PER_CELL)
                if f"{donor}|{session}|{target_name}|draw{d}" in draws_by_key]
        per_session.append(per_session_summary(session, rows))
        all_fitted_k.extend(r["fit"]["k_latents"] for r in rows if r["status"] == "tested")

    with_data = [s for s in per_session if s["n_draws_fitted"] >= 1]
    without_data = [s for s in per_session if s["n_draws_fitted"] == 0]
    a_full_medians = [s["a_full_median"] for s in with_data]
    n = len(with_data)
    n_clear = sum(1 for s in with_data if s["session_clears_own_null_majority"])
    fraction_ci = (list(binomtest(n_clear, n).proportion_ci(1 - ALPHA, method="exact"))
                   if n else [None, None])
    median_a_full = _median_with_ci(a_full_medians, seed_offset)

    k_summary = {
        "k_distribution_over_all_fitted_draws": _status_counts([str(k) for k in all_fitted_k]),
        "fraction_fitted_draws_with_k_equals_8": (float(np.mean([k == 8 for k in all_fitted_k]))
                                                    if all_fitted_k else None),
        "every_fitted_draw_reached_k_equals_8": bool(all_fitted_k and all(k == 8 for k in all_fitted_k)),
    }

    human_median = target_cfg["human_median_a_full"]
    reachable_at_alpha = MIN_ATTAINABLE_PERMUTATION_P <= ALPHA
    if not reachable_at_alpha:
        branch, why = DECLARED_BRANCHES[2], "per-session permutation p at or below alpha is not attainable"
    elif n < MIN_SESSIONS_FOR_PRIMARY_BRANCH:
        branch, why = DECLARED_BRANCHES[2], (
            f"fewer than {MIN_SESSIONS_FOR_PRIMARY_BRANCH} donor sessions produced at least one fitted draw")
    else:
        low, high = median_a_full["ci95"]
        excludes_human = not (low <= human_median <= high)
        frac_low = fraction_ci[0]
        above_half = frac_low is not None and frac_low > 0.5
        if excludes_human and above_half:
            branch, why = DECLARED_BRANCHES[0], "median a_full interval excludes the human median and the clearing fraction interval is above one half"
        elif not excludes_human:
            branch, why = DECLARED_BRANCHES[1], "median a_full interval covers the human arm's median"
        else:
            branch, why = DECLARED_BRANCHES[2], (
                "median a_full interval excludes the human median but the clearing-fraction interval does not "
                "clear one half -- the two pre-declared criteria disagree, so no positive branch is reported")

    return {
        "donor": donor, "target": target_name,
        "match_unit_count": True, "match_trial_count": target_cfg["trial_pool"] is not None,
        "human_dataset": target_cfg["human_dataset"], "human_median_a_full": human_median,
        "human_median_a_full_ci95": target_cfg["human_median_a_full_ci95"],
        "n_donor_sessions_seen": len(donor_sessions),
        "n_donor_sessions_with_at_least_one_fitted_draw": n,
        "n_donor_sessions_excluded_no_fitted_draw": len(without_data),
        "zero_drop_reconciles": bool(len(donor_sessions) == n + len(without_data)),
        "excluded_sessions": [{"session": s["session"], "status_counts": s["status_counts"]} for s in without_data],
        "per_session": per_session,
        "n_sessions_clearing_own_null_majority": n_clear,
        "fraction_clearing_own_null": (n_clear / n) if n else None,
        "fraction_clearing_exact_ci95": fraction_ci,
        "median_a_full": median_a_full,
        "median_a_full_minus_chance": ((median_a_full["median"] - CHANCE_LEVEL)
                                        if median_a_full.get("median") is not None else None),
        "k_summary": k_summary,
        "reachability": {
            "min_attainable_permutation_p": MIN_ATTAINABLE_PERMUTATION_P,
            "reachable_at_alpha": reachable_at_alpha,
            "note": ("CONTENT_N_PERM_FULL is fixed regardless of subsampled trial count, so the "
                     "permutation floor does not depend on the cut; the binding reachability "
                     "constraint here is the donor-session-count floor, checked above."),
        },
        "decision": {
            "declared_branches": list(DECLARED_BRANCHES), "decision_rule": DECISION_RULE,
            "branch": branch, "reason": why,
            "effect_size": {
                "median_a_full": median_a_full.get("median"), "median_a_full_ci95": median_a_full.get("ci95"),
                "human_median_a_full": human_median, "reference_value": human_median,
                "fraction_clearing_own_null": (n_clear / n) if n else None,
                "fraction_clearing_exact_ci95": fraction_ci,
                "n_sessions": n,
            },
        },
    }


# ── Second deliverable: gradient vs ladder, side by side ───────────────────────

def cardinality_gradient_vs_ladder() -> dict:
    human = json.loads(HUMAN_DECODABILITY_PATH.read_text())
    ladder = json.loads(CARDINALITY_LADDER_PATH.read_text())

    between = human["rank_matched_cardinality_description"]["slope_across_preparations"]["fractional_rank"]
    within = ladder["macaque_ladder_slope"]
    split = ladder["split_axis_control"]

    class_range = 8 - 2
    between_prediction = {
        "predictor": "number of classes (linear)",
        "slope": between["slope"], "ci95": between["ci95"], "n_points": between["n_points"],
        "predicted_change_over_2_to_8_classes": between["slope"] * class_range,
        "predicted_change_ci95": [between["ci95"][0] * class_range, between["ci95"][1] * class_range],
    }
    within_prediction = {
        "predictor": within["predictor"],
        "slope": within["mean_slope"], "ci95": [within["ci95_lower"], within["ci95_upper"]],
        "p_value": within["sign_flip_p_value"], "n_sessions": within["n_sessions"],
        "predicted_change_over_2_to_8_classes": within["mean_slope"] * 2,
        "predicted_change_ci95": [within["ci95_lower"] * 2, within["ci95_upper"] * 2],
        "directly_observed_change_two_to_eight_class_rungs": within["fractional_rank_change_from_two_to_eight_classes"],
    }
    return {
        "between_preparation_correlation": between_prediction,
        "within_macaque_causal_manipulation": within_prediction,
        "which_is_causal": (
            "The within-macaque ladder holds preparation, task, recording technology, session set and "
            "estimator fixed and varies only the label's class count; it is the causal test. The "
            "between-preparation gradient covaries class count with species, brain region, session "
            "count, unit count and trial count at once; it is a correlation across preparations, not a "
            "test of class count."
        ),
        "disagreement": (
            "The two disagree in sign and by roughly an order of magnitude over the identical 2-to-8 "
            "class range: the between-preparation slope predicts an increase of about "
            f"{between_prediction['predicted_change_over_2_to_8_classes']:.3f}; the within-macaque "
            "manipulation gives a decrease of about "
            f"{abs(within_prediction['predicted_change_over_2_to_8_classes']):.3f} "
            f"(directly observed rung-to-rung change {within['fractional_rank_change_from_two_to_eight_classes']:.4f}). "
            "Matching label cardinality does not close the mouse-macaque gap; report both numbers, do "
            "not assert a mechanism for the between-preparation gradient, and never report the "
            "within-corpus change as evidence the gradient widens."
        ),
        "two_class_rung_is_axis_dependent": {
            "primary_split_mean_fractional_rank": split["primary_summary"]["mean_fractional_rank"],
            "rotated_split_mean_fractional_rank": split["rotated_summary"]["mean_fractional_rank"],
            "paired_difference": split["paired_mean_difference"],
            "paired_difference_ci95": [split["ci95_lower"], split["ci95_upper"]],
            "p_value": split["p_value"],
            "caveat": ("The two-class rung of the within-macaque ladder takes different values under two "
                       "orthogonal binary splits of the same continuous cue angle; this axis-dependence "
                       "must be carried wherever the two-class rung is quoted, and the claim that survives "
                       "either split is that matched cardinality does not close the gap -- not that it "
                       "widens it."),
        },
    }


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    def log(msg: str) -> None:
        print(f"{msg}  [{time.time() - t0:.1f}s]", file=sys.stderr, flush=True)

    root = data_root()
    targets = load_targets()
    log(f"targets loaded: {list(targets)}")

    mouse_sessions = load_mouse_sessions(root)
    macaque_sessions = load_macaque_sessions(root)
    log(f"donor sessions: mouse {len(mouse_sessions)}, macaque {len(macaque_sessions)}")

    all_sessions = mouse_sessions + macaque_sessions
    draws = load_checkpoint(targets)
    log(f"resumed {len(draws)} draw results from checkpoint")

    expected_keys_per_session = len(targets) * N_DRAWS_PER_SESSION_PER_CELL
    pending = []
    for s in all_sessions:
        keys = {f"{s['donor']}|{s['session']}|{t}|draw{d}" for t in targets for d in range(N_DRAWS_PER_SESSION_PER_CELL)}
        existing = keys & draws.keys()
        if len(existing) < expected_keys_per_session:
            pending.append(s)
    log(f"{len(all_sessions) - len(pending)} of {len(all_sessions)} sessions already fully checkpointed; "
        f"{len(pending)} remaining")

    if pending:
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {}
            for s in pending:
                existing_keys = {f"{s['donor']}|{s['session']}|{t}|draw{d}"
                                  for t in targets for d in range(N_DRAWS_PER_SESSION_PER_CELL)} & draws.keys()
                fut = pool.submit(_session_worker, s["donor"], s["session"], s["counts"], s["labels"],
                                   targets, existing_keys)
                futures[fut] = s
            for fut in as_completed(futures):
                s = futures[fut]
                new_results = fut.result()
                draws.update(new_results)
                save_checkpoint(targets, draws)
                log(f"  {s['donor']}/{s['session']}: {len(new_results)} new draws fitted "
                    f"({len(draws)} total checkpointed)")

    # ── Assemble the artifact ──
    cells = {}
    seed_offset = 0
    for donor, donor_sessions in (("inagaki_alm5", mouse_sessions), ("panichello_2024", macaque_sessions)):
        for target_name, cfg in targets.items():
            cell_key = f"{donor}__{target_name}"
            cells[cell_key] = cell_analysis(donor, target_name, cfg, donor_sessions, draws, seed_offset)
            seed_offset += 1000
            log(f"cell {cell_key}: branch={cells[cell_key]['decision']['branch']} "
                f"median_a_full={cells[cell_key]['median_a_full'].get('median')} "
                f"n={cells[cell_key]['n_donor_sessions_with_at_least_one_fitted_draw']}")

    zero_drop = {
        "n_donor_sessions_seen": {"inagaki_alm5": len(mouse_sessions), "panichello_2024": len(macaque_sessions)},
        "n_cells": len(cells),
        "per_cell_reconciliation": {
            key: {"n_seen": c["n_donor_sessions_seen"],
                  "n_with_data": c["n_donor_sessions_with_at_least_one_fitted_draw"],
                  "n_excluded": c["n_donor_sessions_excluded_no_fitted_draw"],
                  "reconciles": c["zero_drop_reconciles"]}
            for key, c in cells.items()
        },
        "all_cells_reconcile": all(c["zero_drop_reconciles"] for c in cells.values()),
    }

    output = {
        "version": "2026-08-14",
        "status": "complete",
        "scope": {
            "question": (
                "whether the mouse and macaque memorandum-decodability arms still clear the "
                "memorandum's own permutation null after being cut down to a draw from a human "
                "arm's realised unit-count (and, in two of three targets, trial-count) distribution"
            ),
            "estimator": ("session_subtractive_test from scripts/run_state_content_link.py, imported "
                          "unmodified, with k = min(8, n_units - 2, max(2, n_trials // 8)) recomputed "
                          "fresh after every cut"),
            "donors": ["inagaki_alm5", "panichello_2024"],
            "targets": {name: {"unit_pool": cfg["unit_pool"], "trial_pool": cfg["trial_pool"],
                                "human_dataset": cfg["human_dataset"], "n_human_sessions": cfg["n_human_sessions"],
                                "human_median_a_full": cfg["human_median_a_full"]}
                        for name, cfg in targets.items()},
            "n_draws_per_session_per_cell": N_DRAWS_PER_SESSION_PER_CELL,
            "trial_subsampling": "stratified by class, holding the donor session's own class proportions",
            "alpha": ALPHA, "n_bootstrap": N_BOOT, "bootstrap_seed": BOOTSTRAP_SEED,
            "min_sessions_for_primary_branch": MIN_SESSIONS_FOR_PRIMARY_BRANCH,
            "per_draw_seed": "stable checksum of (donor, session, target, draw index)",
            "wall_clock_s": time.time() - t0,
        },
        "cells": cells,
        "cardinality_gradient_vs_within_corpus_ladder": cardinality_gradient_vs_ladder(),
        "zero_drop_accounting": zero_drop,
    }
    _write_json(OUTPUT_PATH, output)
    log(f"wrote {OUTPUT_PATH}")
    print(json.dumps({key: {"branch": c["decision"]["branch"],
                             "median_a_full": c["median_a_full"].get("median"),
                             "n_sessions": c["n_donor_sessions_with_at_least_one_fitted_draw"]}
                       for key, c in cells.items()}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
