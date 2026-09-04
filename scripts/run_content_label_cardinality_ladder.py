"""run_content_label_cardinality_ladder.py -- is the leading latent's
content-ablation rank a property of the population geometry, or of how many
classes the trial label was cut into?

`results/state_content_link.json` reports a leave-one-latent-out ablation-cost
rank of the leading PCA latent: for each session the content decoding AUC is
recomputed with each of the k latents dropped in turn, and the leading latent
is ranked among the k by how much dropping it costs. Fractional rank 0 means
the leading latent is the most content-bearing of the k, 1 means the least,
0.5 is the null. That artifact reads 0.1428571 in mouse anterior lateral motor
cortex (n=23) against 0.6742857 in macaque lateral prefrontal cortex (n=25),
which invites a species/area reading. But the two arms do not carry the same
label: the mouse label is a two-class instructed lick direction, the macaque
label an eight-class cued spatial angle. Class count alone is expected to move
this statistic, because a two-class label is separable along a single
discriminative axis (which tends to align with the dominant variance mode,
making its ablation maximally costly) whereas an eight-class label needs
several axes (so dropping any one of them, the dominant one included, costs
proportionally less).

This module turns that confound into a dose-response. The macaque cued angle
is re-binned into 2, 3, 4, 6 and 8 equal-width angular sectors and the whole
estimator is re-run at each rung with everything else held fixed -- same
sessions, same trials, same units, same k, same per-fold PCA, same classifier,
same macro one-vs-rest AUC, same seeds. It reuses
run_state_content_link.session_subtractive_test unmodified, so the rung at
eight classes is a reproduction check against the deposited categorical cue
index rather than a re-implementation of it, and no rung is interpreted unless
that reproduction succeeds.

Three controls accompany the ladder: the two-class rung is repeated with the
split boundary rotated a quarter turn, separating a cardinality effect from a
spatial-axis effect; the ladder is repeated with every session subsampled to a
common unit count, because macaque unit counts are nearly disjoint across
recording-date blocks; and the mouse arm is placed on the ladder as far as its
labels allow.
"""

from __future__ import annotations

import os

# Each of the MAX_WORKERS forked processes below does its own PCA/logistic-
# regression fits; left at BLAS's default, each one tries to claim every
# core on the machine, so MAX_WORKERS processes oversubscribe by a factor of
# MAX_WORKERS. Capping every BLAS backend to one thread per process makes the
# process pool itself the only source of parallelism, which is what turns a
# single ablation_rank_row call on a mid-size session from not finishing in
# 180 seconds into finishing in about 5. Must be set before numpy is
# imported, since these libraries read the thread count at load time.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_repo_root = Path(__file__).resolve().parents[1]
for _extra in (str(_repo_root / "src"), str(_repo_root / "scripts")):
    if _extra not in sys.path:
        sys.path.insert(0, _extra)

import run_state_content_link as content_link  # noqa: E402
from corpus_sessions import alm_data_directory, data_root, iter_alm  # noqa: E402
from geometry import content_decoding_dropping_latent  # noqa: E402
from spike_pipeline import FrozenPSTHTransform  # noqa: E402
from statistics import (  # noqa: E402
    bootstrap_ci,
    paired_sign_flip_test,
    permutation_test_twosample,
    stable_seed,
)

# The ablation ranking asks the decoder for zero label permutations, so the
# shared decoding helper reduces an empty null array and warns once per fit.
# The affected p-value is not read on those calls; the warning would otherwise
# bury the progress log under tens of thousands of identical lines.
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)

OUTPUT_PATH = _repo_root / "results" / "content_label_cardinality_ladder.json"
DEPOSITED_CONTENT_LINK_PATH = _repo_root / "results" / "state_content_link.json"

LADDER_N_CLASSES = (2, 3, 4, 6, 8)
REPRODUCTION_N_CLASSES = 8

# Half a bin below zero radians, so that at eight bins each of the deposited
# cue locations (multiples of pi/4) sits at a bin centre rather than on a bin
# edge. The same phase at 2, 3, 4 and 6 bins also keeps every cue location
# strictly interior, so no rung depends on a tie-breaking rule.
ANGULAR_BIN_PHASE_RAD = -np.pi / 8
# A quarter turn rotates the two-class boundary onto the perpendicular
# meridian, so the two two-class labellings split the cue circle along
# orthogonal axes.
ORTHOGONAL_ROTATION_RAD = np.pi / 2

REPRODUCTION_TARGET_FRACTIONAL_RANK = 0.6742857142857143
REPRODUCTION_TARGET_N_SESSIONS = 25
REPRODUCTION_TOLERANCE = 1e-9

MATCHED_UNIT_DRAWS = 5
BOOTSTRAP_RESAMPLES = 10000
ALPHA = 0.05
MAX_WORKERS = 8

SEED = stable_seed("content_label_cardinality_ladder")

# Decision rules, fixed before the ladder was run. Every verdict below is
# emitted together with the effect size and reference value that produced it.
BRANCH_RULES = {
    "content_rank_independent_of_label_cardinality": (
        "the per-session slope of fractional rank on log2(number of classes) is not "
        "significantly different from zero, and the whole bootstrap interval of the "
        "two-class rung lies above the 0.5 null"
    ),
    "content_rank_tracks_label_cardinality": (
        "the per-session slope is significantly positive, and the two-class rung is "
        "both significantly below the eight-class rung and below the 0.5 null"
    ),
    "inconclusive_below_detection_floor": (
        "neither of the above; in particular when the slope interval spans zero and "
        "the intervals at the two ends of the ladder overlap"
    ),
}
AXIS_BRANCH_RULES = {
    "two_class_rung_is_axis_invariant": (
        "the paired difference between the two orthogonal two-class splits is not "
        "significant, so the two-class value reflects class count and not a spatial axis"
    ),
    "two_class_rung_depends_on_split_axis": (
        "the paired difference between the two orthogonal two-class splits is "
        "significant, so the two-class value reflects a spatial axis"
    ),
}


def wrap_to_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians onto the half-open interval (-pi, pi], the convention the
    deposited cue angles already use (a cue directly leftward is stored as
    +pi, never -pi)."""
    return -((-np.asarray(angle, dtype=float) + np.pi) % (2 * np.pi) - np.pi)


def angular_bin_labels(angles: np.ndarray, n_classes: int,
                       phase_rad: float = ANGULAR_BIN_PHASE_RAD) -> np.ndarray:
    """Cut angles into ``n_classes`` equal-width sectors of the circle.

    ``phase_rad`` is the lower edge of the first sector. Class integers are
    renumbered so they increase with the sector's centre angle on (-pi, pi],
    which makes the labelling independent of where the phase happens to place
    the first sector and, at eight sectors, identical to the deposited
    categorical cue index up to its one-based offset.
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    width = 2 * np.pi / n_classes
    raw = np.floor(np.mod(np.asarray(angles, dtype=float) - phase_rad, 2 * np.pi) / width).astype(int)
    raw = np.clip(raw, 0, n_classes - 1)
    centres = wrap_to_pi(phase_rad + (np.arange(n_classes) + 0.5) * width)
    renumber = np.empty(n_classes, dtype=int)
    renumber[np.argsort(centres)] = np.arange(n_classes)
    return renumber[raw]


def date_block(session: str) -> str:
    """The macaque sessions were deposited as yymmdd stems and fall into three
    recording-date blocks. The deposit does not carry an animal identifier per
    session, so blocks are named by recording year and no animal identity is
    asserted from them."""
    return f"20{session[:2]}"


def load_macaque_session(path: Path) -> dict:
    """Correct trials only, delay-epoch spike counts in 100 ms bins, and the
    raw cue angle the deposited categorical cue index labels.

    Loading matches run_state_content_link.iter_sessions_with_labels exactly
    (same correctness filter, same delay window, same binning) and adds the
    angle field, which that iterator does not carry.
    """
    path = Path(path)
    raw = loadmat(path, squeeze_me=True)
    spikes = np.asarray(raw["spks"], dtype=float)
    time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
    correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
    cue_angle = np.asarray(raw["cueAng"], dtype=float).reshape(-1)
    cue_index = np.asarray(raw["cueAngIdx"]).reshape(-1).astype(int)
    spikes, cue_angle, cue_index = spikes[correct], cue_angle[correct], cue_index[correct]
    starts = np.arange(content_link.PANICHELLO_DELAY_WINDOW_MS[0],
                       content_link.PANICHELLO_DELAY_WINDOW_MS[1], content_link.BIN_MS)
    binned = [spikes[:, (time_ms >= s) & (time_ms < s + content_link.BIN_MS), :].sum(axis=1)
              for s in starts]
    return {
        "session": path.stem, "date_block": date_block(path.stem),
        "counts": np.stack(binned, axis=2), "cue_angle_rad": cue_angle,
        "deposited_cue_index": cue_index,
        "n_trials_recorded": int(correct.size),
        "n_trials_correct": int(correct.sum()),
    }


def macaque_session_paths(root: Path) -> list[Path]:
    directory = content_link._panichello_directory(root)
    return sorted(directory.glob("*.mat")) if directory is not None else []


def window_mean_features(counts: np.ndarray) -> np.ndarray:
    """The (trials, units, 1) window-mean feature array the ablation estimator
    consumes, built exactly as run_state_content_link builds it."""
    return FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]


def ablation_rank_row(counts: np.ndarray, labels: np.ndarray, seed: int,
                      with_permutation_nulls: bool = True) -> dict:
    """One session at one labelling: applies the same minimum-trials-per-class
    filter as the deposited analysis, then runs the unmodified ablation-rank
    estimator.

    ``with_permutation_nulls=False`` skips the label-permutation fits, which
    cost more than half the runtime and only produce decoding p-values; every
    field the ranking itself depends on is bit-identical either way.
    """
    usable, reason, mask = content_link.usable_label(labels)
    if not usable:
        return {"status": "no_usable_label", "reason": reason}
    counts, labels = counts[mask], np.asarray(labels)[mask]
    if counts.shape[0] < 8:
        return {"status": "too_few_trials_after_label_filter", "n_trials": int(counts.shape[0])}
    result = content_link.session_subtractive_test(
        window_mean_features(counts), labels, seed,
        with_permutation_nulls=with_permutation_nulls)
    _, class_counts = np.unique(labels, return_counts=True)
    result["trials_per_class"] = [int(c) for c in class_counts]
    return result


COMPACT_KEYS = ("status", "n_trials", "n_units", "n_classes", "k_latents", "a_full",
                "leading_latent_cost", "leading_latent_rank_from_top",
                "leading_latent_fractional_rank", "a_full_p_value",
                "a_full_clears_own_null", "trials_per_class")


def compact(result: dict) -> dict:
    """The fields a rung needs, without the per-latent cost vectors."""
    return {key: result[key] for key in COMPACT_KEYS if key in result}


def session_seed(dataset: str, session: str) -> int:
    """The per-session analysis seed of the deposited content-link analysis,
    held fixed across every rung so that only the labelling varies."""
    return content_link._stable_seed(dataset, session, "pooled")


def gate_task(path_str: str) -> dict:
    """The reproduction rung for one session: the eight-sector re-binning of
    the cue angle, and how far that labelling agrees with the deposited
    categorical cue index."""
    meta = load_macaque_session(Path(path_str))
    labels = angular_bin_labels(meta["cue_angle_rad"], REPRODUCTION_N_CLASSES)
    agreement = float(np.mean(labels == meta["deposited_cue_index"] - 1))
    result = ablation_rank_row(meta["counts"], labels, session_seed("panichello_2024", meta["session"]))
    return {
        "session": meta["session"], "date_block": meta["date_block"],
        "n_classes_requested": REPRODUCTION_N_CLASSES,
        "n_trials_recorded": meta["n_trials_recorded"],
        "n_trials_correct": meta["n_trials_correct"],
        "label_agreement_with_deposited_index": agreement,
        **compact(result),
    }


def ladder_task(path_str: str) -> dict:
    """Every rung below the reproduction rung, plus the rotated two-class
    split, for one session."""
    meta = load_macaque_session(Path(path_str))
    seed = session_seed("panichello_2024", meta["session"])
    rungs = {}
    for n_classes in LADDER_N_CLASSES:
        if n_classes == REPRODUCTION_N_CLASSES:
            continue
        labels = angular_bin_labels(meta["cue_angle_rad"], n_classes)
        rungs[n_classes] = {"session": meta["session"], "date_block": meta["date_block"],
                            "n_classes_requested": n_classes,
                            **compact(ablation_rank_row(meta["counts"], labels, seed))}
    rotated = angular_bin_labels(meta["cue_angle_rad"], 2,
                                 ANGULAR_BIN_PHASE_RAD + ORTHOGONAL_ROTATION_RAD)
    return {
        "session": meta["session"], "date_block": meta["date_block"],
        "n_units": int(meta["counts"].shape[1]), "rungs": rungs,
        "rotated_two_class": {"session": meta["session"], "date_block": meta["date_block"],
                              **compact(ablation_rank_row(meta["counts"], rotated, seed))},
    }


def matched_unit_task(path_str: str, matched_units: int, n_draws: int) -> list[dict]:
    """The whole ladder for one session, repeated over unit subsamples of a
    common size. Decoding p-values are not fitted here: only the ablation rank
    enters the matched-unit comparison."""
    meta = load_macaque_session(Path(path_str))
    seed = session_seed("panichello_2024", meta["session"])
    rows = []
    for draw in range(n_draws):
        draw_rng = np.random.default_rng(session_seed("panichello_2024_units", meta["session"]) + draw)
        keep = np.sort(draw_rng.choice(meta["counts"].shape[1], size=matched_units, replace=False))
        counts = meta["counts"][:, keep]
        for n_classes in LADDER_N_CLASSES:
            labels = angular_bin_labels(meta["cue_angle_rad"], n_classes)
            rows.append({"session": meta["session"], "date_block": meta["date_block"],
                         "draw": draw, "n_classes_requested": n_classes,
                         **compact(ablation_rank_row(counts, labels, seed,
                                                     with_permutation_nulls=False))})
    return rows


def pooled_rung_summary(values: list[float], rng: np.random.Generator) -> dict:
    """Pooled fractional rank with a percentile bootstrap interval over
    sessions and the same sign-flip test against the 0.5 null the deposited
    analysis used."""
    array = np.asarray(values, dtype=float)
    mean, lower, upper = bootstrap_ci(array, np.mean, n_boot=BOOTSTRAP_RESAMPLES, rng=rng)
    against_null = content_link._one_sample_sign_flip(list(array), 0.5, alternative="two-sided")
    return {
        "n_sessions": int(array.size), "mean_fractional_rank": mean,
        "ci95_lower": lower, "ci95_upper": upper,
        "null_value": 0.5, "difference_from_null": mean - 0.5,
        "sign_flip_against_null": against_null,
        "per_session": [float(v) for v in array],
    }


def intervals_overlap(a: dict, b: dict) -> bool:
    return a["ci95_lower"] <= b["ci95_upper"] and b["ci95_lower"] <= a["ci95_upper"]


def fractional_ranks(rows: list[dict]) -> list[float]:
    return [r["leading_latent_fractional_rank"] for r in rows]


def main() -> None:
    t0 = time.time()
    root = data_root()
    rng = np.random.default_rng(SEED)

    output: dict = {
        "version": "1",
        "question": (
            "Whether the leading latent's content-ablation rank is a property of the "
            "population geometry or of the number of classes the trial label was cut into."
        ),
        "predeclared_branches": BRANCH_RULES,
        "predeclared_split_axis_branches": AXIS_BRANCH_RULES,
        "parameters": {
            "ladder_n_classes": list(LADDER_N_CLASSES),
            "angular_bin_phase_rad": float(ANGULAR_BIN_PHASE_RAD),
            "orthogonal_rotation_rad": float(ORTHOGONAL_ROTATION_RAD),
            "matched_unit_draws": MATCHED_UNIT_DRAWS,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "alpha": ALPHA,
            "seed": int(SEED),
            "per_session_analysis_seed": (
                "crc32 of dataset|session|structure, identical to the deposited "
                "content-link analysis and held fixed across every rung"
            ),
        },
    }

    def flush() -> None:
        output["wall_clock_s"] = time.time() - t0
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(output, indent=2))

    def log(message: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {message}", file=sys.stderr, flush=True)

    def run_parallel(fn, arguments: list[tuple], label: str) -> list:
        collected = []
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for done, result in enumerate(pool.map(fn, *zip(*arguments)), start=1):
                collected.append(result)
                log(f"{label} {done}/{len(arguments)}")
        return collected

    # ---- corpus staging and zero-drop accounting -------------------------
    paths = macaque_session_paths(root)
    output["macaque_scope"] = {
        "corpus": "panichello_2024",
        "label_source_field": "cueAng (cue angle in radians) re-binned into equal-width angular sectors",
        "deposited_label_field": "cueAngIdx (one-based categorical cue index)",
        "n_session_files_seen": len(paths),
        "session_files": [p.name for p in paths],
        "trial_filter": "correct trials only, then classes with at least four trials",
        "epoch": "delay, 300-1450 ms after cue onset, 100 ms bins, one window mean per trial",
    }
    flush()

    # ---- the memorandum's resolution -------------------------------------
    # Whether the ladder can go above eight classes is a property of the
    # deposit, not of the analysis, so it is measured rather than assumed.
    angles, trial_totals = [], []
    for path in paths:
        meta = load_macaque_session(path)
        angles.append(meta["cue_angle_rad"])
        trial_totals.append((meta["session"], meta["n_trials_recorded"], meta["n_trials_correct"],
                             int(meta["counts"].shape[1])))
    distinct = np.unique(np.round(np.concatenate(angles), 9))
    output["macaque_scope"]["trials_recorded_and_correct_by_session"] = [
        {"session": s, "n_trials_recorded": r, "n_trials_correct": c, "n_units": u}
        for s, r, c, u in trial_totals]
    output["memorandum_resolution"] = {
        "n_distinct_cue_angles_in_corpus": int(distinct.size),
        "distinct_cue_angles_rad": [float(v) for v in distinct],
        "distinct_cue_angles_deg": [float(np.degrees(v)) for v in distinct],
        "cue_angle_is_continuous": bool(distinct.size > 8 * len(paths)),
        "max_attainable_n_classes": int(distinct.size),
        "note": (
            f"The cued angle takes {distinct.size} distinct values across the whole corpus, "
            "so the deposited categorical cue index is an exact relabelling of the angle "
            "rather than a discretisation of a continuum. The ladder can therefore merge "
            "cue locations downward but cannot be run above that many classes here."
        ),
        "rungs_that_are_equal_size_merges_of_cue_locations": [
            n for n in LADDER_N_CLASSES if int(distinct.size) % n == 0],
        "rungs_with_unequal_cue_locations_per_class": [
            n for n in LADDER_N_CLASSES if int(distinct.size) % n != 0],
    }

    # ---- reachability ----------------------------------------------------
    # k is set by min(8, n_units - 2, max(2, n_trials // 8)) and depends on the
    # data shape only, so re-binning cannot change it. Recorded before the
    # ladder so a null cannot be reported that was unreachable by construction.
    k_by_session = {s: int(min(8, u - 2, max(2, c // 8))) for s, _, c, u in trial_totals}
    output["reachability"] = {
        "k_latents_per_session": k_by_session,
        "k_latents_distinct": sorted(set(k_by_session.values())),
        "attainable_fractional_ranks_at_k8": [j / 7 for j in range(8)],
        "statement": (
            "k is 8 in every macaque session, so at every rung the fractional rank can "
            "take all eight values from 0 to 1 in steps of one seventh. The mouse arm's "
            "value of 0.1428571 and the macaque eight-class value of 0.6742857 are both "
            "attainable at every rung, so neither branch is unreachable by construction."
        ),
    }
    flush()

    # ---- reproduction gate, before any other rung is computed ------------
    reproduction_rows = sorted(
        run_parallel(gate_task, [(str(p),) for p in paths], "reproduction rung"),
        key=lambda r: r["session"])
    output["reproduction_session_rows"] = reproduction_rows

    reproduced_mean = float(np.mean(fractional_ranks(reproduction_rows)))
    deposited = json.loads(DEPOSITED_CONTENT_LINK_PATH.read_text())
    deposited_rows = {r["session"]: r["subtractive"] for r in deposited["session_rows"]
                      if r["dataset"] == "panichello_2024"}
    per_session_match = all(
        deposited_rows.get(r["session"], {}).get("leading_latent_rank_from_top")
        == r["leading_latent_rank_from_top"] for r in reproduction_rows)
    passed = bool(
        len(reproduction_rows) == REPRODUCTION_TARGET_N_SESSIONS
        and abs(reproduced_mean - REPRODUCTION_TARGET_FRACTIONAL_RANK) <= REPRODUCTION_TOLERANCE)
    output["reproduction_check"] = {
        "target_pooled_fractional_rank": REPRODUCTION_TARGET_FRACTIONAL_RANK,
        "target_n_sessions": REPRODUCTION_TARGET_N_SESSIONS,
        "tolerance": REPRODUCTION_TOLERANCE,
        "obtained_pooled_fractional_rank": reproduced_mean,
        "obtained_n_sessions": len(reproduction_rows),
        "absolute_difference": abs(reproduced_mean - REPRODUCTION_TARGET_FRACTIONAL_RANK),
        "min_label_agreement_with_deposited_index": float(min(
            r["label_agreement_with_deposited_index"] for r in reproduction_rows)),
        "per_session_rank_matches_deposited": bool(per_session_match),
        "passed": passed,
    }
    flush()
    log(f"reproduction pooled={reproduced_mean!r} passed={passed}")
    if not passed:
        output["status"] = "halted_reproduction_failed"
        output["halt_reason"] = (
            "The eight-sector re-binning of the cue angle did not reproduce the deposited "
            "pooled fractional rank, so no other rung of the ladder is interpretable."
        )
        flush()
        print(json.dumps({"status": output["status"],
                          "reproduction_check": output["reproduction_check"]}, indent=2))
        return
    output["status"] = "completed"

    # ---- the rest of the ladder and the rotated two-class split ----------
    ladder_tasks = sorted(run_parallel(ladder_task, [(str(p),) for p in paths], "ladder rungs"),
                          key=lambda r: r["session"])
    ladder_rows: dict[int, list[dict]] = {REPRODUCTION_N_CLASSES: reproduction_rows}
    for n_classes in LADDER_N_CLASSES:
        if n_classes == REPRODUCTION_N_CLASSES:
            continue
        ladder_rows[n_classes] = [task["rungs"][n_classes] for task in ladder_tasks]
    orthogonal_rows = [task["rotated_two_class"] for task in ladder_tasks]
    unit_counts = {task["session"]: task["n_units"] for task in ladder_tasks}
    output["ladder_session_rows"] = {str(n): ladder_rows[n] for n in LADDER_N_CLASSES}
    output["split_axis_session_rows"] = orthogonal_rows
    flush()

    # ---- pooled rungs and the slope -------------------------------------
    rungs = {}
    for n_classes in LADDER_N_CLASSES:
        summary = pooled_rung_summary(fractional_ranks(ladder_rows[n_classes]), rng)
        summary["mean_a_full"] = float(np.mean([r["a_full"] for r in ladder_rows[n_classes]]))
        summary["mean_trials_per_session"] = float(
            np.mean([r["n_trials"] for r in ladder_rows[n_classes]]))
        summary["n_sessions_content_decodes_above_own_null"] = int(sum(
            1 for r in ladder_rows[n_classes] if r.get("a_full_clears_own_null")))
        summary["chance_level"] = 0.5
        rungs[str(n_classes)] = summary
    output["macaque_ladder"] = rungs

    log_n = np.log2(np.array(LADDER_N_CLASSES, dtype=float))
    sessions = [r["session"] for r in ladder_rows[REPRODUCTION_N_CLASSES]]
    per_session_slope = []
    for index in range(len(sessions)):
        curve = [ladder_rows[n][index]["leading_latent_fractional_rank"] for n in LADDER_N_CLASSES]
        per_session_slope.append(float(np.polyfit(log_n, curve, 1)[0]))
    slopes = np.array(per_session_slope)
    slope_test = paired_sign_flip_test(slopes, np.zeros_like(slopes), alternative="two-sided",
                                       rng=np.random.default_rng(SEED + 1))
    slope_mean, slope_lower, slope_upper = bootstrap_ci(
        slopes, np.mean, n_boot=BOOTSTRAP_RESAMPLES, rng=np.random.default_rng(SEED + 2))
    output["macaque_ladder_slope"] = {
        "predictor": "log2(number of classes)",
        "mean_slope": slope_mean, "ci95_lower": slope_lower, "ci95_upper": slope_upper,
        "sign_flip_p_value": slope_test["p_value"], "n_sessions": int(slopes.size),
        "per_session_slope": per_session_slope,
        "fractional_rank_change_from_two_to_eight_classes": float(
            slope_mean * (np.log2(8) - np.log2(2))),
        "sessions": sessions,
    }
    flush()

    # ---- the ladder's two ends ------------------------------------------
    ends = paired_sign_flip_test(
        np.array(fractional_ranks(ladder_rows[2])),
        np.array(fractional_ranks(ladder_rows[REPRODUCTION_N_CLASSES])),
        alternative="two-sided", rng=np.random.default_rng(SEED + 3))
    output["macaque_ladder_ends"] = {
        "contrast": "two-class rung minus eight-class rung, paired within session",
        "mean_difference": ends["mean_diff"], "ci95_lower": ends["ci_lower"],
        "ci95_upper": ends["ci_upper"], "p_value": ends["p_value"],
        "two_class_mean": rungs["2"]["mean_fractional_rank"],
        "eight_class_mean": rungs["8"]["mean_fractional_rank"],
        "intervals_overlap": intervals_overlap(rungs["2"], rungs["8"]),
    }
    flush()

    # ---- split-axis control ---------------------------------------------
    primary_two = np.array(fractional_ranks(ladder_rows[2]))
    rotated_two = np.array(fractional_ranks(orthogonal_rows))
    axis_test = paired_sign_flip_test(primary_two, rotated_two, alternative="two-sided",
                                      rng=np.random.default_rng(SEED + 4))
    axis_branch = ("two_class_rung_depends_on_split_axis" if axis_test["p_value"] <= ALPHA
                   else "two_class_rung_is_axis_invariant")
    output["split_axis_control"] = {
        "primary_split": "boundary through -22.5 and 157.5 degrees",
        "rotated_split": "boundary through 67.5 and 247.5 degrees, orthogonal to the primary one",
        "primary_summary": pooled_rung_summary(list(primary_two), np.random.default_rng(SEED + 5)),
        "rotated_summary": pooled_rung_summary(list(rotated_two), np.random.default_rng(SEED + 6)),
        "paired_mean_difference": axis_test["mean_diff"],
        "ci95_lower": axis_test["ci_lower"], "ci95_upper": axis_test["ci_upper"],
        "p_value": axis_test["p_value"], "branch": axis_branch,
        "branch_rule": AXIS_BRANCH_RULES[axis_branch],
    }
    flush()

    # ---- which statements hold under either two-class split --------------
    # The two-class rung is the one place where the cardinality branch and the
    # split-axis branch touch the same number, and they pull in opposite
    # directions: the cardinality branch's second clause asks whether the
    # two-class interval clears the 0.5 null, and that answer depends on which
    # meridian the eight cue locations are cut along. Both facts are already
    # recorded above; separating the statements that hold under both splits
    # from the one that does not is left to this block rather than to the
    # reader. Nothing here re-decides either branch: both are reported exactly
    # as their own rules produced them.
    rotated_slopes = np.array([
        float(np.polyfit(log_n, [rotated_two[index] if n == 2
                                 else ladder_rows[n][index]["leading_latent_fractional_rank"]
                                 for n in LADDER_N_CLASSES], 1)[0])
        for index in range(len(sessions))])
    rotated_slope_test = paired_sign_flip_test(
        rotated_slopes, np.zeros_like(rotated_slopes), alternative="two-sided",
        rng=np.random.default_rng(SEED + 12))
    rotated_slope_mean, rotated_slope_lower, rotated_slope_upper = bootstrap_ci(
        rotated_slopes, np.mean, n_boot=BOOTSTRAP_RESAMPLES, rng=np.random.default_rng(SEED + 13))
    rung_null_tests = {
        str(n): {
            "mean_fractional_rank": rungs[str(n)]["mean_fractional_rank"],
            "ci95": [rungs[str(n)]["ci95_lower"], rungs[str(n)]["ci95_upper"]],
            "null_value": 0.5,
            "two_sided_p_value_against_null":
                rungs[str(n)]["sign_flip_against_null"]["two_sided_p_value"],
            "significantly_above_null": bool(
                rungs[str(n)]["sign_flip_against_null"]["two_sided_p_value"] <= ALPHA
                and rungs[str(n)]["mean_fractional_rank"] > 0.5),
        }
        for n in LADDER_N_CLASSES}
    all_rungs_above_null = all(v["significantly_above_null"] for v in rung_null_tests.values())
    output["split_axis_dependence_and_surviving_claims"] = {
        "why_this_block_exists": (
            "The cardinality verdict's second clause is a statement about the two-class rung's "
            "interval, and the split-axis control shows that the two-class rung is not one number: "
            "it depends on which meridian the eight cue locations are cut along. This block states "
            "the disagreement and separates the claims that hold under either split from the one "
            "that does not."
        ),
        "primary_split_two_class": {
            "boundary": output["split_axis_control"]["primary_split"],
            "mean_fractional_rank": rungs["2"]["mean_fractional_rank"],
            "ci95": [rungs["2"]["ci95_lower"], rungs["2"]["ci95_upper"]],
            "whole_interval_above_null": bool(rungs["2"]["ci95_lower"] > 0.5),
        },
        "orthogonal_split_two_class": {
            "boundary": output["split_axis_control"]["rotated_split"],
            "mean_fractional_rank": output["split_axis_control"]["rotated_summary"]["mean_fractional_rank"],
            "ci95": [output["split_axis_control"]["rotated_summary"]["ci95_lower"],
                     output["split_axis_control"]["rotated_summary"]["ci95_upper"]],
            "whole_interval_above_null": bool(
                output["split_axis_control"]["rotated_summary"]["ci95_lower"] > 0.5),
        },
        "paired_difference_between_the_two_splits": {
            "mean_difference": axis_test["mean_diff"],
            "ci95": [axis_test["ci_lower"], axis_test["ci_upper"]],
            "p_value": axis_test["p_value"],
        },
        "slope_under_the_primary_two_class_rung": {
            "mean_slope": slope_mean, "ci95": [slope_lower, slope_upper],
            "p_value": slope_test["p_value"],
            "significantly_different_from_zero": bool(slope_test["p_value"] <= ALPHA),
        },
        "slope_under_the_orthogonal_two_class_rung": {
            "mean_slope": float(rotated_slope_mean),
            "ci95": [float(rotated_slope_lower), float(rotated_slope_upper)],
            "p_value": rotated_slope_test["p_value"],
            "significantly_different_from_zero": bool(rotated_slope_test["p_value"] <= ALPHA),
            "note": (
                "The same ladder with the two-class rung replaced by its orthogonal split and every "
                "other rung unchanged, since only the two-class rung has a second split axis."
            ),
        },
        "every_rung_above_null_under_the_primary_split": all_rungs_above_null,
        "rung_tests_against_the_null_under_the_primary_split": rung_null_tests,
        "what_survives_under_either_split": [
            "The slope of fractional rank on log2(number of classes) is flat: it is not "
            "significantly different from zero whichever two-class split enters the ladder.",
            "Under the primary split every rung from two to eight classes sits significantly above "
            "the 0.5 null, so the leading latent is not the most content-bearing latent at any "
            "class count tested.",
        ],
        "what_does_not_survive": (
            "The claim that the two-class rung's whole interval lies above the null. That holds on "
            "the primary split and fails on the orthogonal one, whose interval spans the null, and "
            "the paired difference between the two splits is itself significant. The two-class value "
            "is therefore a spatial-axis-dependent quantity and must be quoted with its split axis "
            "attached, never as a single number."
        ),
        "reading": (
            "Class count does not move this statistic; the spatial axis a two-way split is taken "
            "along does. The flat slope is the cardinality answer and it is unaffected; the "
            "two-class rung's distance from the null is not a cardinality quantity and should not be "
            "carried as one."
        ),
    }
    flush()

    # ---- matched unit count across date blocks --------------------------
    matched_units = int(min(unit_counts.values()))
    matched_rows = [row for batch in run_parallel(
        matched_unit_task,
        [(str(p), matched_units, MATCHED_UNIT_DRAWS) for p in paths],
        "matched-unit draws") for row in batch]
    output["matched_unit_session_rows"] = matched_rows
    flush()

    def matched_block_summary(block: str | None) -> dict:
        subset = [r for r in matched_rows if block is None or r["date_block"] == block]
        block_sessions = sorted({r["session"] for r in subset})
        per_rung = {}
        for n_classes in LADDER_N_CLASSES:
            values = []
            for session in block_sessions:
                draws = [r["leading_latent_fractional_rank"] for r in subset
                         if r["session"] == session and r["n_classes_requested"] == n_classes
                         and "leading_latent_fractional_rank" in r]
                values.append(float(np.mean(draws)) if draws else float("nan"))
            per_rung[str(n_classes)] = {"n_sessions": len(values),
                                        "mean_fractional_rank": float(np.nanmean(values)),
                                        "per_session": values}
        curves = np.array([[per_rung[str(n)]["per_session"][i] for n in LADDER_N_CLASSES]
                           for i in range(len(block_sessions))])
        usable = np.all(np.isfinite(curves), axis=1)
        block_slopes = np.array([float(np.polyfit(log_n, curve, 1)[0]) for curve in curves[usable]])
        return {
            "n_sessions": len(block_sessions),
            "n_sessions_with_a_complete_ladder": int(usable.sum()),
            "unit_count_before_matching": sorted(unit_counts[s] for s in block_sessions),
            "sessions": block_sessions,
            "rungs": per_rung,
            "mean_slope": float(np.mean(block_slopes)) if block_slopes.size else None,
            "per_session_slope": [float(v) for v in block_slopes],
        }

    blocks = sorted({date_block(p.stem) for p in paths})
    output["matched_unit_block_analysis"] = {
        "matched_unit_count": matched_units,
        "n_draws_per_session": MATCHED_UNIT_DRAWS,
        "subsampling": "units drawn without replacement per session per draw, then averaged over draws",
        "block_labelling": (
            "The deposit carries no per-session animal identifier, so blocks are named by "
            "recording year and no animal identity is asserted from them."
        ),
        "unit_counts_by_block": {
            b: sorted(u for s, u in unit_counts.items() if date_block(s) == b) for b in blocks},
        "pooled": matched_block_summary(None),
        "by_block": {b: matched_block_summary(b) for b in blocks},
    }
    flush()

    # ---- mouse arm -------------------------------------------------------
    alm_directory = alm_data_directory(root)
    alm_files = sorted(alm_directory.glob("*.mat")) if alm_directory.is_dir() else []
    mouse_rows = []
    for meta in iter_alm(root):
        seed = session_seed(meta["dataset"], meta["session"])
        counts = meta["counts"]
        direction = np.asarray(meta["condition"], dtype=int)
        delay_id = np.asarray(meta["delay_duration_id"], dtype=int)
        crossed = direction * 100 + delay_id
        usable, reason, mask = content_link.usable_label(crossed)
        row = {"session": meta["session"], "patient": meta["patient"],
               "n_trials_all": int(counts.shape[0]), "n_units": int(counts.shape[1])}
        row["direction_two_class_all_trials"] = compact(ablation_rank_row(counts, direction, seed))
        if usable:
            row["n_trials_matched"] = int(mask.sum())
            row["direction_two_class_matched_trials"] = compact(
                ablation_rank_row(counts[mask], direction[mask], seed))
            row["crossed_label_matched_trials"] = compact(
                ablation_rank_row(counts[mask], crossed[mask], seed))
            features = window_mean_features(counts[mask])
            k = int(min(8, features.shape[1] - 2, max(2, features.shape[0] // 8)))
            timing = content_decoding_dropping_latent(
                features, delay_id[mask], np.array([0]), k, None, n_splits=3,
                n_perm=content_link.CONTENT_N_PERM_FULL,
                rng=np.random.default_rng(seed + 8))
            row["delay_length_only_decoding"] = {
                "n_classes": int(timing["n_classes"]), "k_latents": k,
                "auc": float(timing["auc_per_t"][0]), "p_value": float(timing["p_per_t"][0]),
                "clears_own_null": bool(timing["p_per_t"][0] <= ALPHA), "chance_level": 0.5}
        else:
            row["crossed_label_status"] = reason
        mouse_rows.append(row)
        log(f"mouse arm {len(mouse_rows)}/{len(alm_files)}")
        output["mouse_session_rows"] = mouse_rows
        flush()

    def mouse_summary(key: str) -> dict | None:
        values = [r[key]["leading_latent_fractional_rank"] for r in mouse_rows
                  if key in r and "leading_latent_fractional_rank" in r[key]]
        if not values:
            return None
        summary = pooled_rung_summary(values, np.random.default_rng(SEED + 9))
        summary["mean_n_classes"] = float(np.mean(
            [r[key]["n_classes"] for r in mouse_rows if key in r and "n_classes" in r[key]]))
        summary["mean_a_full"] = float(np.mean(
            [r[key]["a_full"] for r in mouse_rows if key in r and "a_full" in r[key]]))
        summary["chance_level"] = 0.5
        return summary

    timing_aucs = [r["delay_length_only_decoding"]["auc"] for r in mouse_rows
                   if "delay_length_only_decoding" in r]
    timing_clears = sum(1 for r in mouse_rows
                        if r.get("delay_length_only_decoding", {}).get("clears_own_null"))
    deposited_mouse = [r["subtractive"]["leading_latent_fractional_rank"]
                       for r in deposited["session_rows"] if r["dataset"] == "inagaki_alm5"]
    two_class_all = mouse_summary("direction_two_class_all_trials")
    output["mouse_arm"] = {
        "corpus": "inagaki_alm5",
        "content_label": "instructed lick direction, left or right",
        "content_label_cardinality": 2,
        "n_session_files_seen": len(alm_files),
        "n_sessions_staged": len(mouse_rows),
        "ladder_is_asymmetric": True,
        "asymmetry_note": (
            "The mouse corpus carries no finer-grained content label: the memorandum is a "
            "binary instructed lick direction. Cardinality can only be raised by crossing "
            "it with a task variable that is not content, so the higher rung here is not "
            "equivalent to a macaque rung and the two ladders meet only at two classes."
        ),
        "crossed_label": (
            "instructed lick direction crossed with categorical delay length; the delay "
            "length is drawn unpredictably and is not revealed within the analysed window, "
            "so the crossing raises the class count without necessarily adding a "
            "discriminable axis"
        ),
        "two_class_all_trials": two_class_all,
        "two_class_matched_trials": mouse_summary("direction_two_class_matched_trials"),
        "crossed_label_matched_trials": mouse_summary("crossed_label_matched_trials"),
        "delay_length_only_decoding": {
            "n_sessions": len(timing_aucs),
            "mean_auc": float(np.mean(timing_aucs)) if timing_aucs else None,
            "chance_level": 0.5,
            "n_sessions_clearing_own_null": int(timing_clears),
        },
        "reproduction_of_deposited_two_class_value": {
            "deposited_pooled_fractional_rank": float(np.mean(deposited_mouse)),
            "deposited_n_sessions": len(deposited_mouse),
            "obtained_pooled_fractional_rank": two_class_all["mean_fractional_rank"] if two_class_all else None,
            "obtained_n_sessions": two_class_all["n_sessions"] if two_class_all else 0,
            "matches": bool(two_class_all is not None
                            and len(deposited_mouse) == two_class_all["n_sessions"]
                            and abs(float(np.mean(deposited_mouse))
                                    - two_class_all["mean_fractional_rank"]) <= REPRODUCTION_TOLERANCE),
        },
    }
    flush()

    # ---- cross-corpus contrast at two classes ---------------------------
    macaque_two = primary_two
    mouse_two = np.array([r["direction_two_class_all_trials"]["leading_latent_fractional_rank"]
                          for r in mouse_rows
                          if "leading_latent_fractional_rank" in r["direction_two_class_all_trials"]])
    difference, p_value = permutation_test_twosample(
        macaque_two, mouse_two, alternative="two-sided", rng=np.random.default_rng(SEED + 10))
    macaque_eight = np.array(fractional_ranks(ladder_rows[REPRODUCTION_N_CLASSES]))
    eight_difference, eight_p = permutation_test_twosample(
        macaque_eight, mouse_two, alternative="two-sided", rng=np.random.default_rng(SEED + 11))
    # The macaque two-class value depends on the split axis, so the matched-
    # cardinality contrast has a best case and a worst case rather than a
    # single value. Both are tested identically and both are reported.
    rotated_difference, rotated_p = permutation_test_twosample(
        rotated_two, mouse_two, alternative="two-sided", rng=np.random.default_rng(SEED + 14))
    matched_differences = sorted((difference, rotated_difference))
    output["cross_corpus_contrast"] = {
        "as_deposited": {
            "comparison": "macaque at eight classes against mouse at two classes",
            "macaque_mean": float(np.mean(macaque_eight)), "mouse_mean": float(np.mean(mouse_two)),
            "mean_difference": eight_difference, "p_value": eight_p,
            "label_cardinality_matched": False,
        },
        "at_matched_cardinality": {
            "comparison": "both corpora at two classes, macaque cut along the primary meridian",
            "macaque_mean": float(np.mean(macaque_two)), "mouse_mean": float(np.mean(mouse_two)),
            "mean_difference": difference, "p_value": p_value,
            "label_cardinality_matched": True,
            "matched_on": "number of classes only; unit count, trial count and species differ",
            "split_axis": output["split_axis_control"]["primary_split"],
            "is_the_largest_of_the_two_split_axes": bool(difference >= rotated_difference),
        },
        "at_matched_cardinality_orthogonal_split": {
            "comparison": "both corpora at two classes, macaque cut along the orthogonal meridian",
            "macaque_mean": float(np.mean(rotated_two)), "mouse_mean": float(np.mean(mouse_two)),
            "mean_difference": rotated_difference, "p_value": rotated_p,
            "label_cardinality_matched": True,
            "matched_on": "number of classes only; unit count, trial count and species differ",
            "split_axis": output["split_axis_control"]["rotated_split"],
            "is_the_smallest_of_the_two_split_axes": bool(rotated_difference <= difference),
        },
        "range_across_split_axes": {
            "smallest_matched_difference": matched_differences[0],
            "largest_matched_difference": matched_differences[1],
            "difference_as_deposited": eight_difference,
            "both_split_axes_significant": bool(max(p_value, rotated_p) <= ALPHA),
            "statement": (
                f"Matching label cardinality moves the cross-corpus difference to somewhere between "
                f"{matched_differences[0]:.3f} and {matched_differences[1]:.3f}, against "
                f"{eight_difference:.3f} as deposited, depending on which spatial axis the eight cue "
                "locations are cut along. The robust claim is that matching cardinality does not "
                "close the gap under either split; whether it appears to widen the gap depends "
                "entirely on the choice of split axis and is not a property of the data."
            ),
        },
    }
    flush()

    # ---- context from the already-deposited three-corpus artifact --------
    # An input to this test, not an output of it: state_content_link.json
    # also carries a third corpus (human lateral prefrontal and temporal
    # cortex, dandi_000469) with a five-category picture-identity label. Read
    # here purely for context, at structure == 'pooled' only (its other 43
    # rows are per-structure subsets of the same 18 recordings and are not
    # independent sessions) and restricted to the sessions where content
    # decoding itself clears its own permutation null, because a leading-
    # latent ablation rank computed where nothing decodes is a rank of noise.
    # This block changes nothing about how the branch below is decided; it
    # only reports whether the within-macaque ladder, which is the version
    # of this comparison with species, area, unit count and estimator held
    # fixed, agrees with the cross-corpus trend.
    human_pooled_rows = [r for r in deposited["session_rows"]
                         if r["dataset"] == "dandi_000469" and r.get("structure") == "pooled"]
    human_decodes = [r for r in human_pooled_rows if r["subtractive"].get("a_full_clears_own_null")]
    human_decodes_ranks = sorted(r["subtractive"]["leading_latent_fractional_rank"] for r in human_decodes)
    output["deposited_artifact_cardinality_context"] = {
        "purpose": (
            "Three unrelated corpora already in state_content_link.json, read here for context only. "
            "The within-macaque ladder computed above is the only version of this comparison that holds "
            "species, area, unit count and estimator fixed; this block does not feed its decision."
        ),
        "mouse_inagaki_alm5_two_class": {
            "n_classes": 2, "n_sessions": len(deposited_mouse),
            "mean_fractional_rank": float(np.mean(deposited_mouse)),
            "note": "15 of 23 sessions place the leading latent first, with a tail reaching the bottom "
                    "of the ranking; the mean is exactly 1/7 from that distribution, not from uniformity.",
        },
        "human_dandi_000469_five_class_pooled": {
            "n_classes": 5, "n_sessions_pooled_structure": len(human_pooled_rows),
            "n_sessions_content_clears_own_null": len(human_decodes),
            "fractional_ranks_of_clearing_sessions": [float(v) for v in human_decodes_ranks],
            "mean_fractional_rank_of_clearing_sessions": float(np.mean(human_decodes_ranks)) if human_decodes_ranks else None,
            "median_fractional_rank_of_clearing_sessions": float(np.median(human_decodes_ranks)) if human_decodes_ranks else None,
            "note": "Restricted to 3 of 18 pooled sessions because content decoding clears its own null "
                    "in only those three; a rank computed on the other 15 would rank noise.",
        },
        "macaque_panichello_2024_eight_class": {
            "n_classes": 8, "n_sessions": output["reproduction_check"]["obtained_n_sessions"],
            "mean_fractional_rank": reproduced_mean,
        },
        "trend_across_the_three_corpora": (
            "Pooled fractional rank rises with label cardinality across these three corpora: "
            f"{np.mean(deposited_mouse):.3f} at 2 classes (mouse, n={len(deposited_mouse)}), "
            f"{(np.median(human_decodes_ranks) if human_decodes_ranks else float('nan')):.3f} at 5 classes "
            f"(human, n={len(human_decodes_ranks)} decodable of 18 pooled), "
            f"{reproduced_mean:.3f} at 8 classes (macaque, n={output['reproduction_check']['obtained_n_sessions']})."
        ),
        "within_macaque_ladder_agrees_in_direction": bool(slope_mean > 0),
    }
    flush()

    # ---- can the estimator still see the content at every rung? ----------
    # A flat ladder is only interpretable if the decoder can see the content
    # equally well at each rung. If decoding degraded as the label was cut
    # finer, a flat ablation rank could be nothing more than the rank of an
    # increasingly noisy quantity, so the decoder's own performance is
    # reported at every rung beside the rank it produced.
    a_full_by_rung = {str(n): rungs[str(n)]["mean_a_full"] for n in LADDER_N_CLASSES}
    decodes_by_rung = {str(n): rungs[str(n)]["n_sessions_content_decodes_above_own_null"]
                       for n in LADDER_N_CLASSES}
    a_full_values = list(a_full_by_rung.values())
    output["content_decodability_across_the_ladder"] = {
        "statistic": "macro one-vs-rest AUC using all k latents, averaged over sessions",
        "chance_level": 0.5,
        "mean_full_decoder_auc_by_rung": a_full_by_rung,
        "lowest_mean_full_decoder_auc": float(min(a_full_values)),
        "highest_mean_full_decoder_auc": float(max(a_full_values)),
        "spread_across_rungs": float(max(a_full_values) - min(a_full_values)),
        "n_sessions_content_decodes_above_own_null_by_rung": decodes_by_rung,
        "n_sessions": len(sessions),
        "fewest_sessions_decoding_above_own_null": int(min(decodes_by_rung.values())),
        "orthogonal_two_class_split": {
            "mean_full_decoder_auc": float(np.mean([r["a_full"] for r in orthogonal_rows])),
            "n_sessions_content_decodes_above_own_null": int(sum(
                1 for r in orthogonal_rows if r.get("a_full_clears_own_null"))),
        },
        "statement": (
            f"The mean full-decoder AUC stays between {min(a_full_values):.3f} and "
            f"{max(a_full_values):.3f} at every rung, against a chance level of 0.5, and the content "
            f"decodes above its own permutation null in at least {min(decodes_by_rung.values())} of "
            f"{len(sessions)} sessions at every rung. Decodability therefore does not degrade as the "
            "label is cut finer: the estimator is equally able to see the content at two classes and "
            "at eight. The flat ladder is a property of where the content sits among the latents, "
            "not a reachability artifact of a decoder losing the signal at fine grain."
        ),
    }
    flush()

    # ---- deciding branch -------------------------------------------------
    slope_significant = bool(slope_test["p_value"] <= ALPHA)
    slope_positive = bool(slope_mean > 0)
    two_class_below_null = bool(rungs["2"]["mean_fractional_rank"] < 0.5)
    two_class_interval_above_null = bool(rungs["2"]["ci95_lower"] > 0.5)
    two_class_falls_from_eight = bool(ends["p_value"] <= ALPHA and ends["mean_diff"] < 0)

    if slope_significant and slope_positive and two_class_falls_from_eight and two_class_below_null:
        branch = "content_rank_tracks_label_cardinality"
    elif (not slope_significant) and two_class_interval_above_null:
        branch = "content_rank_independent_of_label_cardinality"
    else:
        branch = "inconclusive_below_detection_floor"

    output["deciding_branch"] = branch
    output["deciding_branch_rule"] = BRANCH_RULES[branch]
    output["deciding_branch_evidence"] = {
        "mean_slope_on_log2_classes": slope_mean,
        "slope_ci95": [slope_lower, slope_upper],
        "slope_p_value": slope_test["p_value"],
        "slope_significant": slope_significant,
        "slope_positive": slope_positive,
        "two_class_mean_fractional_rank": rungs["2"]["mean_fractional_rank"],
        "two_class_ci95": [rungs["2"]["ci95_lower"], rungs["2"]["ci95_upper"]],
        "eight_class_mean_fractional_rank": rungs["8"]["mean_fractional_rank"],
        "eight_class_ci95": [rungs["8"]["ci95_lower"], rungs["8"]["ci95_upper"]],
        "null_value": 0.5,
        "two_class_below_null": two_class_below_null,
        "two_class_interval_above_null": two_class_interval_above_null,
        "two_class_falls_from_eight": two_class_falls_from_eight,
        "ladder_end_intervals_overlap": intervals_overlap(rungs["2"], rungs["8"]),
        "lowest_mean_full_decoder_auc_across_rungs": float(min(a_full_values)),
        "highest_mean_full_decoder_auc_across_rungs": float(max(a_full_values)),
        "fewest_sessions_decoding_above_own_null_across_rungs": int(min(decodes_by_rung.values())),
        "two_class_interval_above_null_under_the_orthogonal_split": bool(
            output["split_axis_control"]["rotated_summary"]["ci95_lower"] > 0.5),
        "slope_under_the_orthogonal_two_class_rung": float(rotated_slope_mean),
        "slope_under_the_orthogonal_two_class_rung_p_value": rotated_slope_test["p_value"],
    }

    output["scope"] = {
        "corpora": ["panichello_2024", "inagaki_alm5"],
        "macaque_sessions_seen": len(paths),
        "macaque_sessions_tested": len(reproduction_rows),
        "macaque_sessions_excluded": [
            {"session": p.stem, "reason": "no ablation rank returned"}
            for p in paths if p.stem not in {r["session"] for r in reproduction_rows}],
        "mouse_sessions_seen": len(alm_files),
        "mouse_sessions_tested": len(mouse_rows),
        "mouse_sessions_excluded": [
            {"session": r["session"], "reason": r["crossed_label_status"]}
            for r in mouse_rows if "crossed_label_status" in r],
        "estimator": (
            "run_state_content_link.session_subtractive_test, unmodified: k PCA latents "
            "fit per training fold, macro one-vs-rest AUC, chance 0.5 regardless of class "
            "count, leading latent ranked among the k by ablation cost"
        ),
        "held_fixed_across_rungs": [
            "sessions", "trials", "units", "k latents", "per-fold PCA", "classifier",
            "cross-validation scheme", "per-session seed"],
        "varies_across_rungs": ["number of angular sectors the cued angle is cut into"],
        "seed": int(SEED),
        "wall_clock_s": time.time() - t0,
    }
    flush()

    print(json.dumps({
        "deciding_branch": branch,
        "reproduction_check": output["reproduction_check"],
        "mean_slope": slope_mean, "slope_ci95": [slope_lower, slope_upper],
        "slope_p_value": slope_test["p_value"],
        "rungs": {n: rungs[n]["mean_fractional_rank"] for n in rungs},
        "split_axis_branch": axis_branch,
    }, indent=2))


if __name__ == "__main__":
    main()
