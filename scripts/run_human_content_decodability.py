"""run_human_content_decodability.py -- is the memorandum decodable from the human
population state at all, and only then, where does the leading variance mode sit in
the ranking of latents by decoding cost?

The delivered content-link artifact (results/state_content_link.json, read-only here
and never recomputed or edited) reports one number per (dataset, structure, session):
the median macro one-vs-rest AUC of a content decoder built on k PCA latents of the
delay-epoch population window mean, ``a_full``, its own label-permutation p-value, and
the leading latent's fractional rank among all k latents ranked by leave-one-out
ablation cost. Read across corpora it says the leading variance mode carries the
memorandum in mouse anterior lateral motor cortex and does not in macaque lateral
prefrontal cortex, with a human depth-electrode corpus reported as a null in between.

That reading rests on three properties of the human arm that the delivered artifact
records but does not act on, each re-verified directly against it here:

  Pseudo-replication on one side only. The human corpus dandi_000469 contributes 61
  rows from 18 recording sessions -- one pooled-structure row per session plus 43
  per-anatomical-structure rows that are subsets of the same recordings -- while the
  mouse and macaque arms are pooled-only, 23 rows from 23 sessions and 25 from 25.
  Every arm here is therefore restricted to ``structure == 'pooled'``, one row per
  recording, and nothing downstream ever sees a per-structure row.

  A statistic that is undefined where nothing decodes. The fractional rank ranks
  latents by how much removing each one costs the decoder. Where the full decoder does
  not beat its own label-permutation null, those costs are differences between noise
  draws and their ranking carries no content information. In dandi_000469 the full
  decoder clears its own null in 3 of 18 pooled sessions, against 23 of 23 in the
  mouse arm and 24 of 25 in the macaque arm, so a rank pooled over that corpus is
  mostly ranking noise.

  An unmatched rank parameter. The estimator sets k = min(8, n_units - 2,
  max(2, n_trials // 8)). In dandi_000469 the trial count binds (30-45 usable trials)
  and k comes out 3, 4 or 5, so the fractional rank -- which divides by k - 1 -- is a
  three-to-five-valued statistic on a coarse grid, while both animal arms sit at k = 8.

This module therefore does three things the delivered artifact does not.

Decodability first, as its own result. Per corpus, at pooled structure only, it reports
the fraction of sessions in which ``a_full`` clears its own permutation null, the median
``a_full`` with a percentile bootstrap confidence interval, and the distance above the
chance level of 0.5. Macro one-vs-rest AUC has chance 0.5 whatever the class count, so
this one number is comparable across arms whose label cardinality differs; that is the
property that makes it the right primary quantity here.

A second human arm at the animal arms' rank parameter. The corpus dandi_001187 is
absent from the delivered artifact entirely -- 76 rows excluded as
``fewer_than_2_classes_with_at_least_4_trials`` against the label field
``PicIDs_Encoding1``. The field is present and populated; each picture is shown exactly
once per session, so at image-identity grain every session yields zero classes with four
or more trials by design. The grain fails, not the data. ``PicIDs_Encoding1 // 100``
gives five picture-category codes, the same grain the sibling corpus dandi_000469
already uses, and with roughly 130 correct trials per session the trial term stops
binding k. This module adds that arm at category grain and reports the realised k per
session rather than assuming it.

A bound on the human null instead of a bare non-significant p-value. For each human arm
it reports the smallest departure of the fractional rank from 0.5 the arm could have
detected at 80% power, and the arm's power against each animal arm's observed effect,
so that "the human arm is null" is resolvable into which effect sizes it excludes.

The estimator itself is imported unmodified from the module that produced the delivered
artifact, so the new arm is measured with the same instrument as the old ones, and a
reproduction gate re-runs every pooled session of dandi_000469 through that same call
path and requires agreement with the delivered numbers before any dependent analysis is
allowed to run.
"""

from __future__ import annotations

import os

# BLAS thread caps must precede the first numpy import: several analyses run
# concurrently on this machine and an uncapped thread pool per process turns
# small matrix work into scheduler contention.
for _thread_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from corpus_sessions import (  # noqa: E402
    EPOCH_WINDOWS_S,
    MIN_TRIALS,
    MIN_UNITS_POOLED,
    data_root,
    iter_dandi_000469,
    region_filtered_units,
)
from run_human_drift_spine_001187_000673 import _trial_group, canonical_sessions  # noqa: E402
from run_state_content_link import (  # noqa: E402
    CONTENT_N_PERM_FULL,
    MIN_CLASSES,
    MIN_TRIALS_PER_CLASS,
    _one_sample_sign_flip,
    _stable_seed,
    delay_counts,
    session_subtractive_test,
    usable_label,
)
from spike_pipeline import FrozenPSTHTransform, load_spike_times, resolve_unit_regions  # noqa: E402
from state_persistence import _ols_slope  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci,
    fdr_bh,
    minimum_detectable_paired_difference,
    power_to_detect_effect,
)

CONTENT_LINK_PATH = REPO_ROOT / "results" / "state_content_link.json"
OUTPUT_PATH = REPO_ROOT / "results" / "human_content_decodability.json"
CHECKPOINT_PATH = REPO_ROOT / "results" / ".checkpoints" / "human_content_decodability_sessions.json"

ALPHA = 0.05
CHANCE_LEVEL = 0.5
NULL_FRACTIONAL_RANK = 0.5
STRUCTURE_OF_ANALYSIS = "pooled"
N_BOOT = 2000
BOOTSTRAP_SEED = 20260814

# dandi_001187 ships one picture identifier per trial and each picture is shown once
# per session; the hundreds digit is the picture's semantic category, which is the
# grain at which the sibling corpus dandi_000469 already carries a usable label.
PICTURE_CATEGORY_DIVISOR = 100
EXPECTED_PICTURE_CATEGORIES = 5

# Pre-declared decision rules. Each threshold below was fixed before any output of this
# module was inspected, and none of them may be moved to make a result come out.
MIN_SESSIONS_FOR_PRIMARY_BRANCH = 8
MIN_SESSIONS_FOR_CONDITIONED_RANK = 4
REPRODUCTION_EXACT_TOLERANCE = 0.0
REPRODUCTION_NUMERIC_TOLERANCE = 1e-6

PRIMARY_BRANCHES = (
    "memorandum_decodable_in_the_human_population_state",
    "memorandum_not_decodable_in_the_human_population_state",
    "inconclusive_below_detection_floor",
)
SECONDARY_BRANCHES = (
    "leading_latent_carries_the_content",
    "leading_latent_does_not_carry_the_content",
    "leading_latent_rank_not_separated_from_the_null",
    "content_rank_undefined_at_this_decodability",
)
REPRODUCTION_BRANCHES = (
    "reproduced_exactly",
    "reproduced_within_numeric_tolerance",
    "reproduction_failed",
)

PRIMARY_DECISION_RULE = (
    "At pooled structure only. Reachability first: a per-session permutation p of at "
    "most 0.05 must be attainable given the null replicate count, otherwise "
    "inconclusive_below_detection_floor. Then, with fewer than "
    f"{MIN_SESSIONS_FOR_PRIMARY_BRANCH} sessions, inconclusive_below_detection_floor. "
    "Otherwise the exact (Clopper-Pearson) 95% interval on the fraction of sessions "
    "whose full-latent decoder clears its own null decides: if that interval covers "
    "0.5 the arm cannot distinguish a majority from a minority and the branch is "
    "inconclusive_below_detection_floor; if the fraction is above 0.5 the branch is "
    "memorandum_decodable_in_the_human_population_state; otherwise it is "
    "memorandum_not_decodable_in_the_human_population_state."
)
SECONDARY_DECISION_RULE = (
    "At pooled structure only, on the sessions whose full-latent decoder clears its own "
    f"null. With fewer than {MIN_SESSIONS_FOR_CONDITIONED_RANK} such sessions, or where "
    "the sign-flip test reports itself underpowered by construction, the branch is "
    "content_rank_undefined_at_this_decodability. Otherwise one Benjamini-Hochberg pass "
    "is applied across the conditioned two-sided sign-flip tests of all four arms, and "
    "an arm whose corrected q is at most 0.05 fires "
    "leading_latent_carries_the_content when its conditioned mean fractional rank is "
    "below 0.5 and leading_latent_does_not_carry_the_content when it is above; an arm "
    "whose q exceeds 0.05 fires leading_latent_rank_not_separated_from_the_null."
)

ARM_DESCRIPTIONS = {
    "inagaki_alm5": {
        "preparation": "mouse anterior lateral motor cortex, silicon-probe single units",
        "label": "instructed lick direction",
    },
    "panichello_2024": {
        "preparation": "macaque lateral prefrontal cortex, array single units",
        "label": "cued angular position bin",
    },
    "dandi_000469": {
        "preparation": "human medial temporal and frontal depth-electrode single units",
        "label": "encoded picture category",
    },
    "dandi_001187": {
        "preparation": "human medial temporal and frontal depth-electrode single units",
        "label": "encoded picture category",
    },
}
HUMAN_ARMS = ("dandi_000469", "dandi_001187")


def _json_default(obj):
    """Fallback encoder that never raises. A run that has already paid for its
    computation must not lose it to one unserialisable field, so unknown types
    degrade to their repr rather than aborting the write."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default))
    tmp.replace(path)


def picture_category_labels(picture_ids: np.ndarray) -> np.ndarray:
    """Semantic category code of each encoded picture."""
    return np.asarray(picture_ids, dtype=int) // PICTURE_CATEGORY_DIVISOR


def class_counts(labels: np.ndarray) -> dict[int, int]:
    classes, counts = np.unique(np.asarray(labels), return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}


# ── Session-level computation, checkpointed ───────────────────────────────────

def _estimator_fingerprint() -> dict:
    """Everything that would invalidate a cached session result if it changed."""
    return {
        "label_grain": "picture_identifier_integer_divided_by_%d" % PICTURE_CATEGORY_DIVISOR,
        "structure": STRUCTURE_OF_ANALYSIS,
        "min_classes": MIN_CLASSES,
        "min_trials_per_class": MIN_TRIALS_PER_CLASS,
        "n_permutations_full_model": CONTENT_N_PERM_FULL,
    }


def load_checkpoint() -> dict:
    if not CHECKPOINT_PATH.exists():
        return {}
    try:
        cached = json.loads(CHECKPOINT_PATH.read_text())
    except json.JSONDecodeError:
        return {}
    if cached.get("estimator_fingerprint") != _estimator_fingerprint():
        return {}
    return cached.get("sessions", {})


def save_checkpoint(sessions: dict) -> None:
    _write_json(CHECKPOINT_PATH, {
        "estimator_fingerprint": _estimator_fingerprint(),
        "n_sessions": len(sessions),
        "sessions": sessions,
    })


def iter_dandi_001187_pooled_with_exclusions(root: Path):
    """Every canonical dandi_001187 session with its pooled-structure admission
    status, so that sessions the shared loader silently declines to yield are
    still accounted for. Applies exactly the shared loader's gates -- the same
    trial floor, the same region filter and firing-rate quality control, the same
    pooled unit floor -- and additionally reports which gate each declined session
    failed, which the shared loader has no way to express."""
    for meta in canonical_sessions():
        if meta["primary_release"] != "001187":
            continue
        path = root / meta["primary_path"]
        row = {"dataset": "dandi_001187", "patient": meta["patient"], "session": Path(meta["primary_path"]).stem}
        if not path.exists():
            yield {**row, "admission": "deposited_file_absent"}
            continue
        with h5py.File(path, "r") as handle:
            if "units" not in handle:
                yield {**row, "admission": "no_spike_sorted_units_table"}
                continue
            spike_lists_all = load_spike_times(handle)
            unit_regions = resolve_unit_regions(handle)["region"]
            trials = _trial_group(handle, "001187")
            accuracy = trials["response_accuracy"][:].astype(bool)
            picture_ids = trials["PicIDs_Encoding1"][:].astype(int)
            t_maint = trials["timestamps_Maintenance"][:]
        n_correct = int(accuracy.sum())
        if n_correct < MIN_TRIALS:
            yield {**row, "admission": "fewer_correct_trials_than_minimum",
                   "n_correct_trials": n_correct, "minimum_trials": MIN_TRIALS}
            continue
        onsets = t_maint[accuracy]
        spike_lists = region_filtered_units(
            spike_lists_all, unit_regions, STRUCTURE_OF_ANALYSIS, onsets, EPOCH_WINDOWS_S["delay"])
        ids = picture_ids[accuracy]
        label_census = {
            "n_correct_trials": n_correct,
            "n_distinct_picture_identifiers": int(len(np.unique(ids))),
            "picture_identity_grain_class_counts": class_counts(ids),
            "picture_category_grain_class_counts": class_counts(picture_category_labels(ids)),
        }
        if spike_lists is None:
            yield {**row, "admission": "fewer_pooled_units_than_minimum",
                   "minimum_units": MIN_UNITS_POOLED, **label_census}
            continue
        yield {**row, "admission": "admitted", "spike_lists": spike_lists, "onsets": onsets,
               "picture_ids": ids, **label_census}


def subtractive_row(dataset: str, session: str, spike_lists, onsets: np.ndarray,
                    labels: np.ndarray, window_s: float) -> dict:
    """One session through the unmodified content-link estimator."""
    ok, reason, mask = usable_label(labels)
    if not ok:
        return {"status": "no_usable_label", "reason": reason}
    counts = delay_counts(spike_lists, onsets, window_s)[mask]
    kept_labels = np.asarray(labels)[mask]
    window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
    seed = _stable_seed(dataset, session, STRUCTURE_OF_ANALYSIS)
    subtractive = session_subtractive_test(window_mean, kept_labels, seed)
    return {
        "status": subtractive.get("status", "unknown"),
        "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
        "class_counts": class_counts(kept_labels), "seed": int(seed),
        "subtractive": subtractive,
    }


def compute_dandi_001187_arm(root: Path, checkpoint: dict, log) -> tuple[list[dict], list[dict]]:
    """Pooled-structure rows of dandi_001187 at picture-category grain, plus one
    census row per canonical session including the declined ones."""
    tested, census = [], []
    for entry in iter_dandi_001187_pooled_with_exclusions(root):
        session = entry["session"]
        census_row = {k: v for k, v in entry.items() if k not in ("spike_lists", "onsets", "picture_ids")}
        if entry["admission"] != "admitted":
            census.append(census_row)
            log(f"  dandi_001187/{session}: {entry['admission']}")
            continue

        key = f"dandi_001187|{session}|{STRUCTURE_OF_ANALYSIS}"
        if key in checkpoint:
            row = checkpoint[key]
        else:
            labels = picture_category_labels(entry["picture_ids"])
            row = subtractive_row("dandi_001187", session, entry["spike_lists"], entry["onsets"],
                                  labels, EPOCH_WINDOWS_S["delay"])
            row = {"dataset": "dandi_001187", "patient": entry["patient"], "session": session,
                   "structure": STRUCTURE_OF_ANALYSIS, "label_field": "PicIDs_Encoding1_category", **row}
            checkpoint[key] = row
            save_checkpoint(checkpoint)

        census.append({**census_row, "estimator_status": row["status"]})
        if row["status"] != "tested":
            log(f"  dandi_001187/{session}: estimator status {row['status']}")
            continue
        tested.append(row)
        s = row["subtractive"]
        log(f"  dandi_001187/{session}: k={s['k_latents']} a_full={s['a_full']:.4f} "
            f"p={s['a_full_p_value']:.3f} rank={s['leading_latent_rank_from_top']}")
    return tested, census


# ── Reproduction gate ─────────────────────────────────────────────────────────

GATE_FIELDS = ("a_full", "a_full_p_value", "k_latents", "leading_latent_fractional_rank")


def reproduction_gate(root: Path, deposited_rows: list[dict], checkpoint: dict, log) -> dict:
    """Re-runs every pooled session of dandi_000469 through this module's own call
    path and compares against the delivered content-link artifact. Nothing that
    depends on the new arm may be believed unless the same estimator, driven the
    same way, returns the delivered numbers."""
    deposited = {r["session"]: r for r in deposited_rows if r["dataset"] == "dandi_000469"}
    comparisons, recomputed = [], []
    for meta in iter_dandi_000469(root):
        if meta["structure"] != STRUCTURE_OF_ANALYSIS:
            continue
        session = meta["session"]
        key = f"dandi_000469|{session}|{STRUCTURE_OF_ANALYSIS}"
        if key in checkpoint:
            row = checkpoint[key]
        else:
            row = subtractive_row("dandi_000469", session, meta["spike_lists"],
                                  meta["epoch_onsets"]["delay"], np.asarray(meta["item_ids"]),
                                  meta["epoch_windows"]["delay"])
            row = {"dataset": "dandi_000469", "patient": meta["patient"], "session": session,
                   "structure": STRUCTURE_OF_ANALYSIS, "label_field": meta["item_id_field"], **row}
            checkpoint[key] = row
            save_checkpoint(checkpoint)
        recomputed.append(row)
        reference = deposited.get(session)
        if reference is None or row["status"] != "tested":
            comparisons.append({"session": session, "status": "no_delivered_counterpart_or_not_tested"})
            continue
        diffs = {f: float(row["subtractive"][f]) - float(reference["subtractive"][f]) for f in GATE_FIELDS}
        comparisons.append({"session": session, "status": "compared", "absolute_differences":
                            {f: abs(d) for f, d in diffs.items()}})
        log(f"  gate dandi_000469/{session}: max |difference| "
            f"{max(abs(d) for d in diffs.values()):.3e}")

    compared = [c for c in comparisons if c["status"] == "compared"]
    max_abs = max((max(c["absolute_differences"].values()) for c in compared), default=float("inf"))
    if not compared:
        branch = "reproduction_failed"
    elif max_abs <= REPRODUCTION_EXACT_TOLERANCE:
        branch = "reproduced_exactly"
    elif max_abs <= REPRODUCTION_NUMERIC_TOLERANCE:
        branch = "reproduced_within_numeric_tolerance"
    else:
        branch = "reproduction_failed"
    return {
        "declared_branches": list(REPRODUCTION_BRANCHES),
        "branch": branch,
        "fields_compared": list(GATE_FIELDS),
        "n_sessions_recomputed": len(recomputed),
        "n_sessions_compared": len(compared),
        "n_sessions_delivered": len(deposited),
        "max_absolute_difference": max_abs,
        "exact_tolerance": REPRODUCTION_EXACT_TOLERANCE,
        "numeric_tolerance": REPRODUCTION_NUMERIC_TOLERANCE,
        "per_session": comparisons,
    }


# ── Arm-level statistics ──────────────────────────────────────────────────────

def _values(rows: list[dict], field: str) -> list[float]:
    return [float(r["subtractive"][field]) for r in rows]


def _median_with_ci(values: list[float], seed_offset: int) -> dict:
    if not values:
        return {"status": "not_computable", "n": 0}
    if len(values) < 2:
        return {"status": "point_estimate_only", "n": 1, "median": float(values[0])}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    observed, low, high = bootstrap_ci(np.asarray(values, dtype=float), np.median, n_boot=N_BOOT, rng=rng)
    return {"status": "computed", "n": len(values), "median": observed,
            "ci95": [low, high], "n_boot": N_BOOT}


def _mean_with_ci(values: list[float], seed_offset: int) -> dict:
    if not values:
        return {"status": "not_computable", "n": 0}
    if len(values) < 2:
        return {"status": "point_estimate_only", "n": 1, "mean": float(values[0])}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    observed, low, high = bootstrap_ci(np.asarray(values, dtype=float), np.mean, n_boot=N_BOOT, rng=rng)
    return {"status": "computed", "n": len(values), "mean": observed,
            "ci95": [low, high], "n_boot": N_BOOT}


def arm_decodability(dataset: str, rows: list[dict]) -> dict:
    """The primary quantity: does the memorandum decode from the population state
    at all, in how many sessions, and by how much above chance."""
    a_full = _values(rows, "a_full")
    p_values = _values(rows, "a_full_p_value")
    clearing = [p <= ALPHA for p in p_values]
    n, n_clear = len(rows), int(sum(clearing))
    median_a_full = _median_with_ci(a_full, seed_offset=_stable_seed(dataset) % 1000)
    fraction_ci = (list(binomtest(n_clear, n).proportion_ci(1 - ALPHA, method="exact"))
                   if n else [None, None])
    min_attainable_p = 1.0 / (CONTENT_N_PERM_FULL + 1)
    distance = None
    if median_a_full.get("status") == "computed":
        distance = {
            "median_minus_chance": median_a_full["median"] - CHANCE_LEVEL,
            "ci95_minus_chance": [median_a_full["ci95"][0] - CHANCE_LEVEL,
                                  median_a_full["ci95"][1] - CHANCE_LEVEL],
        }
    return {
        "dataset": dataset,
        "preparation": ARM_DESCRIPTIONS.get(dataset, {}).get("preparation"),
        "label": ARM_DESCRIPTIONS.get(dataset, {}).get("label"),
        "structure": STRUCTURE_OF_ANALYSIS,
        "n_sessions": n,
        "n_sessions_clearing_own_null": n_clear,
        "fraction_clearing_own_null": (n_clear / n) if n else None,
        "fraction_clearing_exact_ci95": [float(x) for x in fraction_ci] if n else None,
        "median_a_full": median_a_full,
        "distance_above_chance": distance,
        "chance_level": CHANCE_LEVEL,
        "chance_level_note": "macro one-vs-rest AUC; chance is 0.5 regardless of class count",
        "median_a_full_p_value": float(np.median(p_values)) if p_values else None,
        "min_attainable_permutation_p": min_attainable_p,
        "permutation_p_reachable_at_alpha": bool(min_attainable_p <= ALPHA),
        "realised_k_distribution": class_counts(_values(rows, "k_latents")),
        "n_classes_distribution": class_counts(_values(rows, "n_classes")),
        "n_units_median": float(np.median([r["n_units"] for r in rows])) if rows else None,
        "n_units_range": [int(min(r["n_units"] for r in rows)), int(max(r["n_units"] for r in rows))] if rows else None,
        "n_trials_median": float(np.median([r["n_trials"] for r in rows])) if rows else None,
        "n_trials_range": [int(min(r["n_trials"] for r in rows)), int(max(r["n_trials"] for r in rows))] if rows else None,
    }


def primary_branch(arm: dict) -> dict:
    n, n_clear = arm["n_sessions"], arm["n_sessions_clearing_own_null"]
    if not arm["permutation_p_reachable_at_alpha"]:
        branch, why = PRIMARY_BRANCHES[2], "a per-session permutation p at or below alpha is not attainable"
    elif n < MIN_SESSIONS_FOR_PRIMARY_BRANCH:
        branch, why = PRIMARY_BRANCHES[2], f"fewer than {MIN_SESSIONS_FOR_PRIMARY_BRANCH} pooled sessions"
    else:
        low, high = arm["fraction_clearing_exact_ci95"]
        if low <= 0.5 <= high:
            branch, why = PRIMARY_BRANCHES[2], "the exact interval on the clearing fraction covers one half"
        elif n_clear / n > 0.5:
            branch, why = PRIMARY_BRANCHES[0], "the clearing fraction is above one half and its interval excludes it"
        else:
            branch, why = PRIMARY_BRANCHES[1], "the clearing fraction is below one half and its interval excludes it"
    return {
        "declared_branches": list(PRIMARY_BRANCHES),
        "decision_rule": PRIMARY_DECISION_RULE,
        "branch": branch,
        "reason": why,
        "effect_size": {
            "fraction_clearing_own_null": arm["fraction_clearing_own_null"],
            "fraction_clearing_exact_ci95": arm["fraction_clearing_exact_ci95"],
            "n_sessions_clearing_own_null": n_clear,
            "n_sessions": n,
            "median_a_full": arm["median_a_full"].get("median"),
            "median_a_full_ci95": arm["median_a_full"].get("ci95"),
            "median_a_full_minus_chance": (arm["distance_above_chance"] or {}).get("median_minus_chance"),
            "reference_value": CHANCE_LEVEL,
        },
    }


def arm_fractional_rank(dataset: str, rows: list[dict]) -> dict:
    """The leading latent's ablation-cost rank, unconditioned and conditioned on the
    session's own decoder clearing its own null. Conditioning selects on whether the
    statistic is defined -- whether there is any content signal for an ablation to
    cost -- not on the value the statistic takes; the selection check beside it is
    what makes that claim checkable rather than asserted."""
    offset = _stable_seed(dataset) % 1000
    conditioned = [r for r in rows if bool(r["subtractive"]["a_full_clears_own_null"])]
    dropped = [r for r in rows if not bool(r["subtractive"]["a_full_clears_own_null"])]
    out = {"dataset": dataset, "n_sessions": len(rows),
           "n_sessions_retained_by_conditioning": len(conditioned),
           "n_sessions_lost_to_conditioning": len(dropped)}
    for name, subset, off in (("unconditioned", rows, offset), ("conditioned", conditioned, offset + 500)):
        values = _values(subset, "leading_latent_fractional_rank") if subset else []
        out[name] = {
            "n_sessions": len(subset),
            "mean_fractional_rank": _mean_with_ci(values, off),
            "mean_minus_null": (float(np.mean(values)) - NULL_FRACTIONAL_RANK) if values else None,
            "null_value": NULL_FRACTIONAL_RANK,
            "sign_flip_test_two_sided": _one_sample_sign_flip(values, NULL_FRACTIONAL_RANK, alternative="two-sided"),
            "per_session_values": values,
        }
    out["selection_check"] = _selection_check(conditioned, dropped)
    return out


def _selection_check(retained: list[dict], dropped: list[dict]) -> dict:
    """Does conditioning on decodability move the arm in the direction of the effect,
    or only remove sessions where the statistic has nothing to measure?"""
    def _describe(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        ranks = _values(subset, "leading_latent_fractional_rank")
        return {
            "n": len(subset),
            "mean_fractional_rank": float(np.mean(ranks)),
            "median_fractional_rank": float(np.median(ranks)),
            "mean_a_full": float(np.mean(_values(subset, "a_full"))),
            "median_n_units": float(np.median([r["n_units"] for r in subset])),
            "median_n_trials": float(np.median([r["n_trials"] for r in subset])),
            "median_k_latents": float(np.median(_values(subset, "k_latents"))),
        }
    a, b = _describe(retained), _describe(dropped)
    difference = None
    if a.get("n") and b.get("n"):
        difference = {
            "mean_fractional_rank_retained_minus_dropped": a["mean_fractional_rank"] - b["mean_fractional_rank"],
            "median_n_units_retained_minus_dropped": a["median_n_units"] - b["median_n_units"],
            "median_n_trials_retained_minus_dropped": a["median_n_trials"] - b["median_n_trials"],
        }
    return {
        "what_the_filter_selects_on": (
            "whether the session's full-latent decoder beats its own label-permutation "
            "null, which is the condition under which a leave-one-latent-out ablation "
            "cost has any content signal to remove; it does not select on the leading "
            "latent's rank among those costs"
        ),
        "retained": a, "dropped": b, "difference": difference,
    }


def secondary_branches(rank_blocks: dict) -> dict:
    """One Benjamini-Hochberg pass across the conditioned rank tests of every arm,
    then a branch per arm. Correcting arms of unequal session count separately is
    what manufactures dissociations between them."""
    testable = [(ds, blk) for ds, blk in rank_blocks.items()
                if blk["conditioned"]["sign_flip_test_two_sided"].get("status") == "tested"
                and blk["n_sessions_retained_by_conditioning"] >= MIN_SESSIONS_FOR_CONDITIONED_RANK]
    q_by_dataset = {}
    if testable:
        p = [blk["conditioned"]["sign_flip_test_two_sided"]["p_value"] for _, blk in testable]
        corrected = fdr_bh(np.asarray(p, dtype=float), alpha=ALPHA)
        q_by_dataset = {ds: float(q) for (ds, _), q in zip(testable, corrected["q_values"])}

    branches = {}
    for dataset, block in rank_blocks.items():
        conditioned = block["conditioned"]
        test = conditioned["sign_flip_test_two_sided"]
        q = q_by_dataset.get(dataset)
        if q is None:
            branch = SECONDARY_BRANCHES[3]
            why = (f"fewer than {MIN_SESSIONS_FOR_CONDITIONED_RANK} sessions clear the decodability "
                   f"primary, or the sign-flip test reports itself underpowered by construction")
        elif q <= ALPHA and conditioned["mean_fractional_rank"].get("mean", 1.0) < NULL_FRACTIONAL_RANK:
            branch, why = SECONDARY_BRANCHES[0], "conditioned mean fractional rank below the null after correction"
        elif q <= ALPHA:
            branch, why = SECONDARY_BRANCHES[1], "conditioned mean fractional rank above the null after correction"
        else:
            branch, why = SECONDARY_BRANCHES[2], "conditioned mean fractional rank not separated from the null after correction"
        branches[dataset] = {
            "declared_branches": list(SECONDARY_BRANCHES),
            "decision_rule": SECONDARY_DECISION_RULE,
            "branch": branch, "reason": why,
            "effect_size": {
                "conditioned_mean_fractional_rank": conditioned["mean_fractional_rank"].get("mean"),
                "conditioned_mean_fractional_rank_ci95": conditioned["mean_fractional_rank"].get("ci95"),
                "conditioned_mean_minus_null": conditioned["mean_minus_null"],
                "null_value": NULL_FRACTIONAL_RANK,
                "n_sessions_conditioned": block["n_sessions_retained_by_conditioning"],
                "n_sessions_lost_to_conditioning": block["n_sessions_lost_to_conditioning"],
                "uncorrected_two_sided_p": test.get("p_value"),
                "corrected_q": q,
                "unconditioned_mean_fractional_rank": block["unconditioned"]["mean_fractional_rank"].get("mean"),
            },
        }
    return {"per_arm": branches, "n_tests_in_correction_family": len(testable),
            "correction": "Benjamini-Hochberg across the conditioned two-sided sign-flip tests of every arm"}


# ── Bounding the human null ───────────────────────────────────────────────────

def detectability_bound(dataset: str, rank_block: dict, reference_effects: dict) -> dict:
    """The smallest departure of the fractional rank from its null that this arm could
    have detected at 80% power, and the arm's power against each animal arm's observed
    departure. The design is one sample per session against a fixed null value, not a
    paired difference; the same spread-and-count arithmetic applies, with the paired
    difference replaced by the per-session departure from the null."""
    out = {"dataset": dataset,
           "design_note": ("one value per session tested against a fixed null of "
                           f"{NULL_FRACTIONAL_RANK}, so the paired-difference bound is applied to the "
                           "per-session departure from that null rather than to a within-session "
                           "difference between two conditions"),
           "reference_effects": reference_effects}
    for name in ("unconditioned", "conditioned"):
        departures = [v - NULL_FRACTIONAL_RANK for v in rank_block[name]["per_session_values"]]
        bound = minimum_detectable_paired_difference(departures) if len(departures) >= 2 else {
            "status": "not_computable", "n": len(departures),
            "reason": "fewer than 2 sessions -- no spread to estimate"}
        power = {}
        for ref_name, ref_value in reference_effects.items():
            power[ref_name] = (power_to_detect_effect(ref_value, departures)
                               if len(departures) >= 2 else {"status": "not_computable", "n": len(departures)})
        excludes = {}
        if bound.get("status") == "computed":
            for ref_name, ref_value in reference_effects.items():
                excludes[ref_name] = bool(abs(ref_value) >= bound["mdd"])
        out[name] = {
            "n_sessions": len(departures),
            "observed_mean_departure": float(np.mean(departures)) if departures else None,
            "minimum_detectable_departure_at_80_percent_power": bound,
            "power_against_reference_effects": power,
            "reference_effect_is_at_or_above_the_detectable_bound": excludes,
        }
    return out


# ── Rank-matched cardinality description ──────────────────────────────────────

def cardinality_points(arms: dict, rank_blocks: dict, arm_rows: dict, matched_k: int) -> dict:
    """The three preparations whose realised rank parameter is the same, placed side by
    side against their label cardinality."""
    included, excluded = {}, {}
    for dataset, arm in arms.items():
        realised = arm["realised_k_distribution"]
        if list(realised) == [matched_k] and len(class_counts_keys(arm["n_classes_distribution"])) == 1:
            included[dataset] = arm
        else:
            excluded[dataset] = {"realised_k_distribution": realised,
                                 "n_classes_distribution": arm["n_classes_distribution"],
                                 "reason": "realised rank parameter or label cardinality is not single-valued at the matched rank"}
    points = []
    for dataset, arm in included.items():
        n_classes = int(class_counts_keys(arm["n_classes_distribution"])[0])
        block = rank_blocks[dataset]
        points.append({
            "dataset": dataset,
            "preparation": arm["preparation"], "label": arm["label"],
            "n_classes": n_classes, "k_latents": matched_k,
            "n_sessions": arm["n_sessions"],
            "fraction_clearing_own_null": arm["fraction_clearing_own_null"],
            "median_a_full": arm["median_a_full"].get("median"),
            "median_a_full_ci95": arm["median_a_full"].get("ci95"),
            "mean_fractional_rank": block["unconditioned"]["mean_fractional_rank"].get("mean"),
            "mean_fractional_rank_ci95": block["unconditioned"]["mean_fractional_rank"].get("ci95"),
            "n_units_median": arm["n_units_median"], "n_trials_median": arm["n_trials_median"],
        })
    points.sort(key=lambda p: p["n_classes"])

    slopes = {}
    if len(points) >= 2:
        per_arm_x = {p["dataset"]: p["n_classes"] for p in points}
        for field, label in (("leading_latent_fractional_rank", "fractional_rank"), ("a_full", "a_full")):
            xs, ys, groups = [], [], []
            for dataset in per_arm_x:
                for row in arm_rows[dataset]:
                    xs.append(per_arm_x[dataset])
                    ys.append(float(row["subtractive"][field]))
                    groups.append(dataset)
            slopes[label] = _stratified_slope_ci(np.asarray(xs, float), np.asarray(ys, float), groups)
    return {
        "matched_k_latents": matched_k,
        "points": points,
        "arms_excluded_from_the_matched_comparison": excluded,
        "slope_across_preparations": slopes,
        "what_this_comparison_cannot_separate": (
            "Species, brain region, task, recording technology, session count, unit count and trial "
            "count all change together across these three preparations. Matching the rank parameter "
            "removes one of those confounds and leaves every other one standing, so this is a "
            "description of three preparations at a common rank, not a test of label cardinality. "
            "Testing cardinality requires holding the preparation fixed and varying only the number "
            "of classes, which only a ladder run inside a single corpus can do."
        ),
    }


def class_counts_keys(counts: dict) -> list:
    return sorted(int(float(k)) for k in counts)


def _stratified_slope_ci(x: np.ndarray, y: np.ndarray, groups: list[str]) -> dict:
    """Ordinary least-squares slope of a per-session value on label cardinality, with a
    percentile interval from resampling sessions within each preparation, so the
    interval reflects the sessions available and not a pretence of exchangeability
    between preparations."""
    observed = _ols_slope(x, y)
    if observed is None:
        return {"status": "not_computable", "reason": "label cardinality carries no variance across the arms"}
    index_by_group: dict[str, np.ndarray] = {}
    for g in sorted(set(groups)):
        index_by_group[g] = np.array([i for i, gg in enumerate(groups) if gg == g])
    rng = np.random.default_rng(BOOTSTRAP_SEED + 7)
    draws = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in index_by_group.values()])
        s = _ols_slope(x[idx], y[idx])
        if s is not None:
            draws.append(s)
    if not draws:
        return {"status": "not_computable", "slope": observed}
    low, high = np.percentile(draws, [2.5, 97.5])
    return {"status": "computed", "slope": observed, "ci95": [float(low), float(high)],
            "n_boot": len(draws), "n_points": int(len(x)),
            "resampling": "sessions resampled with replacement within each preparation"}


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()
    # content_decoding_dropping_latent takes the mean of an empty permutation array
    # whenever it is called for a cost ranking only; the resulting value is never read.
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")

    def log(message: str) -> None:
        print(f"{message}  [{time.time() - t0:.1f}s]", file=sys.stderr, flush=True)

    root = data_root()
    delivered = json.loads(CONTENT_LINK_PATH.read_text())
    delivered_pooled = [r for r in delivered["session_rows"] if r["structure"] == STRUCTURE_OF_ANALYSIS]
    checkpoint = load_checkpoint()
    log(f"resumed {len(checkpoint)} session results from the checkpoint")

    gate = reproduction_gate(root, delivered_pooled, checkpoint, log)
    log(f"reproduction gate: {gate['branch']} (max |difference| {gate['max_absolute_difference']:.3e})")
    if gate["branch"] == "reproduction_failed":
        _write_json(OUTPUT_PATH, {
            "version": "2026-08-14", "status": "halted_at_reproduction_gate",
            "reproduction_gate": gate, "wall_clock_s": time.time() - t0,
            "note": ("The estimator did not return the delivered numbers when driven from this "
                     "module, so no dependent analysis was run."),
        })
        raise SystemExit(f"reproduction gate failed: max absolute difference {gate['max_absolute_difference']}")

    new_rows, census_001187 = compute_dandi_001187_arm(root, checkpoint, log)

    arm_rows: dict[str, list[dict]] = {
        dataset: [r for r in delivered_pooled if r["dataset"] == dataset]
        for dataset in ("inagaki_alm5", "panichello_2024", "dandi_000469")
    }
    arm_rows["dandi_001187"] = new_rows

    arms = {ds: arm_decodability(ds, rows) for ds, rows in arm_rows.items()}
    primary = {ds: primary_branch(arm) for ds, arm in arms.items()}
    rank_blocks = {ds: arm_fractional_rank(ds, rows) for ds, rows in arm_rows.items()}
    secondary = secondary_branches(rank_blocks)

    reference_effects = {
        "mouse_anterior_lateral_motor_cortex_observed_departure":
            rank_blocks["inagaki_alm5"]["unconditioned"]["mean_minus_null"],
        "macaque_lateral_prefrontal_cortex_observed_departure":
            rank_blocks["panichello_2024"]["unconditioned"]["mean_minus_null"],
    }
    bounds = {ds: detectability_bound(ds, rank_blocks[ds], reference_effects) for ds in HUMAN_ARMS}

    matched = cardinality_points(arms, rank_blocks, arm_rows, matched_k=8)

    admissions = class_counts_from_strings([c["admission"] for c in census_001187])
    estimator_statuses = class_counts_from_strings(
        [c["estimator_status"] for c in census_001187 if "estimator_status" in c])
    n_admitted = admissions.get("admitted", 0)
    zero_drop = {
        "dandi_001187": {
            "n_canonical_sessions_seen": len(census_001187),
            "n_admitted_at_pooled_structure": n_admitted,
            "n_tested": len(new_rows),
            "admission_counts": admissions,
            "estimator_status_counts_among_admitted": estimator_statuses,
            "reconciliation_holds": bool(
                len(census_001187) == sum(admissions.values())
                and n_admitted == sum(estimator_statuses.values())
                and len(new_rows) == estimator_statuses.get("tested", 0)),
            "label_grain_note": (
                "Every canonical session carries a populated picture identifier per trial, and at "
                "picture-identity grain none of them yields a class with enough trials because each "
                "picture is shown once. The sessions counted out below are counted out by the shared "
                "loader's minimum trial count and minimum pooled unit count, not by the label."
            ),
            "per_session": census_001187,
        },
        "previously_excluded_rows_now_reclassified": reclassify_delivered_exclusions(delivered, census_001187),
        "delivered_arms": {
            ds: {"n_rows_delivered_all_structures":
                 sum(1 for r in delivered["session_rows"] if r["dataset"] == ds),
                 "n_rows_used_at_pooled_structure": len(arm_rows[ds]),
                 "n_rows_dropped_as_per_structure_subsets":
                 sum(1 for r in delivered["session_rows"] if r["dataset"] == ds) - len(arm_rows[ds])}
            for ds in ("inagaki_alm5", "panichello_2024", "dandi_000469")
        },
    }

    output = {
        "version": "2026-08-14",
        "status": "complete",
        "scope": {
            "question": ("whether the memorandum is decodable from the delay-epoch population state at "
                         "all in each preparation, and only where it is, where the leading variance "
                         "mode sits among the latents ranked by leave-one-out ablation cost"),
            "unit_of_analysis": ("one recording session at structure == 'pooled'; per-anatomical-"
                                 "structure rows are subsets of the same recordings and are excluded "
                                 "from every arm"),
            "estimator": ("session_subtractive_test from scripts/run_state_content_link.py, imported "
                          "unmodified, with k = min(8, n_units - 2, max(2, n_trials // 8)) PCA latents "
                          "on the delay-epoch window mean"),
            "corpora": sorted(arm_rows),
            "label_grain_dandi_001187": ("PicIDs_Encoding1 // %d, the encoded picture's semantic "
                                         "category" % PICTURE_CATEGORY_DIVISOR),
            "correct_trials_only_note": ("The dandi_001187 loader keeps correct trials only, so no "
                                         "correct-versus-error contrast is available in that corpus."),
            "alpha": ALPHA, "n_bootstrap": N_BOOT, "bootstrap_seed": BOOTSTRAP_SEED,
            "per_session_seed": "stable checksum of (dataset, session, structure)",
            "wall_clock_s": time.time() - t0,
        },
        "reproduction_gate": gate,
        "primary_decodability": {"per_arm": arms, "branches": primary},
        "secondary_fractional_rank": {"per_arm": rank_blocks, "branches": secondary},
        "human_null_bounds": bounds,
        "rank_matched_cardinality_description": matched,
        "zero_drop_accounting": zero_drop,
        "session_rows_dandi_001187": new_rows,
    }
    _write_json(OUTPUT_PATH, output)
    log(f"wrote {OUTPUT_PATH}")
    print(json.dumps({
        "reproduction_gate": gate["branch"],
        "primary": {ds: {"branch": b["branch"],
                         "fraction_clearing": arms[ds]["fraction_clearing_own_null"],
                         "median_a_full": arms[ds]["median_a_full"].get("median")}
                    for ds, b in primary.items()},
        "secondary": {ds: b["branch"] for ds, b in secondary["per_arm"].items()},
    }, indent=2, default=_json_default))


def class_counts_from_strings(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))


def reclassify_delivered_exclusions(delivered: dict, census: list[dict]) -> dict:
    """Every row the delivered content-link artifact excluded for dandi_001187, with the
    status it takes at picture-category grain."""
    by_session = {c["session"]: c for c in census}
    rows = []
    for row in delivered["status_rows"]:
        if row["dataset"] != "dandi_001187":
            continue
        census_row = by_session.get(row["session"])
        if row["structure"] != STRUCTURE_OF_ANALYSIS:
            new_status = "out_of_scope_analysis_is_pooled_structure_only"
        elif census_row is None:
            new_status = "not_reached_by_the_shared_loader"
        elif census_row["admission"] != "admitted":
            new_status = census_row["admission"]
        else:
            new_status = census_row.get("estimator_status", "unknown")
        rows.append({"session": row["session"], "structure": row["structure"],
                     "delivered_status": row.get("reason") or row.get("status"),
                     "status_at_picture_category_grain": new_status})
    return {
        "n_rows": len(rows),
        "status_counts": class_counts_from_strings([r["status_at_picture_category_grain"] for r in rows]),
        "rows": rows,
    }


if __name__ == "__main__":
    main()
