"""run_state_content_link.py -- does the cross-unit state r_lag_profile
measures carry item content, or is it a shared slow signal beside the
content code?

Five rounds have measured this state without asking what it carries. The
test here is subtractive, not comparative: on the population's window-mean
features (k PCA latents, held-out trials only), content decoding accuracy
with the leading latent (the same direction state_persistence.py's
r_lag_profile tracks) projected out is compared against accuracy with a
COMPARABLE latent projected out instead, within session, pooled across
sessions by a sign-flip test on the leading latent's rank among all k by
decoding cost. A trial-resolved companion splits held-out trials by the
content decoder's own out-of-fold margin and asks whether d_perm (Section
1's observable) is larger where content is more decodable, at matched trial
count. Reuses geometry.content_decoding_dropping_latent (itself built on the
existing PCA-per-fold/classifier/permutation-null stack),
geometry.out_of_fold_class_confidence, state_persistence.r_lag_profile and
per_unit_permutation_null_r_lag_profile, and src/corpus_sessions for scope.
"""

from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_src_dir = str(Path(__file__).resolve().parents[1] / "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from corpus_sessions import data_root, iter_alm, iter_all_corpora  # noqa: E402
from geometry import content_decoding_dropping_latent, out_of_fold_class_confidence  # noqa: E402
from spike_pipeline import FrozenPSTHTransform, build_psth  # noqa: E402
from state_persistence import _ols_slope, per_unit_permutation_null_r_lag_profile, r_lag_profile  # noqa: E402
from statistics import paired_sign_flip_test  # noqa: E402
from info_decoding import MIN_CLASSES  # noqa: E402
from spike_pipeline import DECIDING_WIDTH_BINS, delay_counts  # noqa: E402
from statistics import CONTENT_N_PERM_FULL, PANICHELLO_DELAY_WINDOW_MS, _one_sample_sign_flip, _stable_seed  # noqa: E402
from corpus_sessions import _panichello_directory  # noqa: E402
from spike_pipeline import BIN_MS  # noqa: E402
from info_decoding import session_subtractive_test  # noqa: E402

BIN_WIDTH_S = BIN_MS / 1000.0
MIN_TRIALS_PER_CLASS = 4
MIN_TRIALS_FOR_MARGIN_SPLIT = 16
MARGIN_SUBSAMPLE_REPEATS = 8
MARGIN_MATCHED_N_SPLITS = 8
MARGIN_MATCHED_N_NULL_REPLICATES = 10

LAG_SHAPE_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_shape.json"
LAG_PATH = Path(__file__).resolve().parents[1] / "results" / "state_persistence_lag.json"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "results" / "state_content_link.json"




def usable_label(labels: np.ndarray | None, min_classes: int = MIN_CLASSES, min_per_class: int = MIN_TRIALS_PER_CLASS) -> tuple[bool, str | None, np.ndarray | None]:
    if labels is None:
        return False, "no_label_field", None
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    eligible = classes[counts >= min_per_class]
    if len(eligible) < min_classes:
        return False, f"fewer_than_{min_classes}_classes_with_at_least_{min_per_class}_trials", None
    mask = np.isin(labels, eligible)
    return True, None, mask






def session_trial_resolved_test(counts: np.ndarray, X: np.ndarray, labels: np.ndarray, k: int,
                                 clearing_lags: list[int], breakpoint_bins: int, seed: int) -> dict:
    """Splits held-out trials by the content decoder's own out-of-fold
    margin (probability assigned to the true class) into high/low halves,
    then recomputes d_perm at the deciding width within each half at a
    MATCHED trial count (both halves subsampled to the smaller half's size,
    repeated draws, distribution reported rather than one draw). A narrower
    scope than the full trial-resolved battery this could in principle
    support, documented rather than silent: only the margin split is run
    (not a separate correct/incorrect split), and the late segment is not
    recomputed per subset -- both dropped for compute-budget reasons."""
    if counts.shape[0] < MIN_TRIALS_FOR_MARGIN_SPLIT:
        return {"status": "too_few_trials_to_split", "n_trials": int(counts.shape[0])}

    conf_rng = np.random.default_rng(seed)
    confidence = np.asarray(out_of_fold_class_confidence(X, labels, t_idx=0, n_components=k, n_splits=5, rng=conf_rng)).reshape(-1)
    order = np.argsort(confidence)
    half = len(order) // 2
    low_idx, high_idx = order[:half], order[half:]
    if len(low_idx) < 8 or len(high_idx) < 8:
        return {"status": "too_few_trials_per_margin_half", "n_low": int(len(low_idx)), "n_high": int(len(high_idx))}

    n_target = min(len(low_idx), len(high_idx))

    def _draw_summary(pool: np.ndarray, repeat: int) -> dict | None:
        rng = np.random.default_rng(seed + 4000 + repeat)
        sub = rng.choice(pool, size=n_target, replace=False) if len(pool) > n_target else pool
        sub_counts = counts[sub]
        profile = r_lag_profile(sub_counts, DECIDING_WIDTH_BINS, n_splits=MARGIN_MATCHED_N_SPLITS, rng=rng)
        if profile["status"] != "fitted":
            return None
        null = per_unit_permutation_null_r_lag_profile(
            sub_counts, DECIDING_WIDTH_BINS, n_replicates=MARGIN_MATCHED_N_NULL_REPLICATES, rng=rng)
        lags_present = sorted(set(profile["lags"]) & set(null["lags"]))
        if not lags_present:
            return None
        d_perm = {lag: profile["lags"][lag]["r_median"] - null["lags"][lag]["r_null_median"] for lag in lags_present}
        common_clearing = [lag for lag in clearing_lags if lag in d_perm]
        d_perm_mean_clearing = float(np.mean([d_perm[lag] for lag in common_clearing])) if common_clearing else None
        early_lags = [lag for lag in lags_present if lag <= breakpoint_bins]
        early_slope = None
        if len(early_lags) >= 2:
            early_slope = _ols_slope([lag * BIN_WIDTH_S for lag in early_lags], [d_perm[lag] for lag in early_lags])
        return {"d_perm_mean_over_clearing_lags": d_perm_mean_clearing, "early_slope": early_slope}

    low_draws, high_draws = [], []
    for r in range(MARGIN_SUBSAMPLE_REPEATS):
        low_result = _draw_summary(low_idx, r)
        high_result = _draw_summary(high_idx, r + 1000)
        if low_result is not None:
            low_draws.append(low_result)
        if high_result is not None:
            high_draws.append(high_result)

    def _summary(draws: list[dict], key: str) -> dict:
        vals = [d[key] for d in draws if d.get(key) is not None]
        if not vals:
            return {"status": "not_computed"}
        return {"status": "tested", "n_draws": len(vals), "median": float(np.median(vals)),
                "iqr": [float(np.percentile(vals, 25)), float(np.percentile(vals, 75))]}

    paired_diff = [
        h["d_perm_mean_over_clearing_lags"] - l["d_perm_mean_over_clearing_lags"]
        for h, l in zip(high_draws, low_draws)
        if h.get("d_perm_mean_over_clearing_lags") is not None and l.get("d_perm_mean_over_clearing_lags") is not None
    ]

    return {
        "status": "tested", "n_target_trials_per_half": int(n_target),
        "high_margin_d_perm": _summary(high_draws, "d_perm_mean_over_clearing_lags"),
        "low_margin_d_perm": _summary(low_draws, "d_perm_mean_over_clearing_lags"),
        "high_margin_early_slope": _summary(high_draws, "early_slope"),
        "low_margin_early_slope": _summary(low_draws, "early_slope"),
        "median_paired_diff_high_minus_low": float(np.median(paired_diff)) if paired_diff else None,
        "n_paired_draws": len(paired_diff),
    }


HUMAN_DATASETS = ("dandi_000469", "dandi_001187", "dandi_000574")






def iter_sessions_with_labels(root: Path):
    """Yields (dataset, patient, session, structure, spike_lists_or_counts_ready,
    onset, window_s, labels, label_field) for every (dataset, structure,
    session) this module's content-link scope reaches. Human corpora: delay
    epoch, 100 ms bins, item identity if the corpus carries it (see
    src/corpus_sessions.py's item_ids field per iterator). ALM: instructed
    lick direction. Macaque spatial working-memory corpus: cue angle index (8-way)."""
    for meta in iter_all_corpora(root):
        yield {
            "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
            "structure": meta["structure"], "spike_lists": meta["spike_lists"],
            "onset": meta["epoch_onsets"]["delay"], "window_s": meta["epoch_windows"]["delay"],
            "labels": meta.get("item_ids"), "label_field": meta.get("item_id_field"),
            "label_unavailable_reason": meta.get("item_id_unavailable_reason"),
        }

    for meta in iter_alm(root, bin_ms=BIN_MS, window_s=2.0):
        counts = meta["counts"]
        yield {
            "dataset": meta["dataset"], "patient": meta["patient"], "session": meta["session"],
            "structure": meta["structure"], "counts_precomputed": counts,
            "labels": meta.get("condition"), "label_field": meta.get("item_id_field"),
            "label_unavailable_reason": None,
        }

    directory = _panichello_directory(root)
    if directory is not None:
        for path in sorted(glob.glob(str(directory / "*.mat"))):
            raw = loadmat(path, squeeze_me=True)
            spikes = np.asarray(raw["spks"], dtype=float)
            time_ms = np.asarray(raw["tc"], dtype=float).reshape(-1)
            correct = np.asarray(raw["isCorr"], dtype=bool).reshape(-1)
            cue_idx = np.asarray(raw["cueAngIdx"]).reshape(-1)
            spikes, cue_idx = spikes[correct], cue_idx[correct]
            starts = np.arange(PANICHELLO_DELAY_WINDOW_MS[0], PANICHELLO_DELAY_WINDOW_MS[1], BIN_MS)
            binned = [spikes[:, (time_ms >= s) & (time_ms < s + BIN_MS), :].sum(axis=1) for s in starts]
            counts = np.stack(binned, axis=2)  # (trial, unit, bin)
            yield {
                "dataset": "panichello_2024", "patient": Path(path).stem, "session": Path(path).stem,
                "structure": "pooled", "counts_precomputed": counts,
                "labels": cue_idx.astype(int), "label_field": "cueAngIdx (8-way cued color/location bin)",
                "label_unavailable_reason": None,
            }




def main() -> None:
    t0 = time.time()
    root = data_root()

    lag = json.loads(LAG_PATH.read_text())
    shape = json.loads(LAG_SHAPE_PATH.read_text())
    clearing_lags = lag["clearing_lags_bins_at_deciding_width"]
    breakpoint_bins = shape["arms"]["human_delay"]["by_width"][str(DECIDING_WIDTH_BINS)]["breakpoint_summary"]["declared_breakpoint_bins"]

    rows = []
    status_rows = []
    n_sessions_seen = 0
    for entry in iter_sessions_with_labels(root):
        n_sessions_seen += 1
        ok, reason, mask = usable_label(entry.get("labels"))
        base = {"dataset": entry["dataset"], "patient": entry["patient"], "session": entry["session"],
                "structure": entry.get("structure", "pooled"), "label_field": entry.get("label_field")}
        if not ok:
            status_rows.append({**base, "status": "no_usable_label",
                                 "reason": entry.get("label_unavailable_reason") or reason})
            continue

        if "counts_precomputed" in entry:
            counts_full = entry["counts_precomputed"]
        else:
            counts_full = delay_counts(entry["spike_lists"], entry["onset"], entry["window_s"])
        labels_full = np.asarray(entry["labels"])
        counts, labels = counts_full[mask], labels_full[mask]
        if counts.shape[0] < 8:
            status_rows.append({**base, "status": "too_few_trials_after_label_filter", "n_trials": int(counts.shape[0])})
            continue

        window_mean = FrozenPSTHTransform().fit(counts).transform(counts).mean(axis=2)[:, :, None]
        seed = _stable_seed(entry["dataset"], entry["session"], entry.get("structure", "pooled"))
        subtractive = session_subtractive_test(window_mean, labels, seed)
        if subtractive.get("status") != "tested":
            status_rows.append({**base, "status": subtractive.get("status"), "detail": subtractive})
            continue
        trial_resolved = session_trial_resolved_test(
            counts, window_mean, labels, subtractive["k_latents"], clearing_lags, breakpoint_bins, seed)

        rows.append({**base, "n_trials": int(counts.shape[0]), "n_units": int(counts.shape[1]),
                     "subtractive": subtractive, "trial_resolved": trial_resolved})
        print(f"  {entry['dataset']}/{entry['session']}/{entry.get('structure','pooled')}: "
              f"k={subtractive['k_latents']} a_full={subtractive['a_full']:.3f} "
              f"rank={subtractive['leading_latent_rank_from_top']} "
              f"elapsed={time.time()-t0:.1f}s", file=sys.stderr)

    # 1B.3: pooled deciding test on the fractional rank, overall and per corpus.
    def _pool(rows_subset: list[dict]) -> dict:
        fractional_ranks = [r["subtractive"]["leading_latent_fractional_rank"] for r in rows_subset]
        rank_test = _one_sample_sign_flip(fractional_ranks, 0.5, alternative="less")
        a_full_ps = [r["subtractive"]["a_full_p_value"] for r in rows_subset]
        n_clearing = sum(1 for p in a_full_ps if p <= 0.05)
        diffs = [r["trial_resolved"]["median_paired_diff_high_minus_low"] for r in rows_subset
                 if r["trial_resolved"].get("status") == "tested" and r["trial_resolved"].get("median_paired_diff_high_minus_low") is not None]
        trial_resolved_test = _one_sample_sign_flip(diffs, 0.0, alternative="greater") if diffs else {"status": "not_computed"}
        return {
            "n_sessions": len(rows_subset), "fractional_rank_test": rank_test,
            "n_sessions_a_full_clears_own_null": n_clearing, "n_sessions_a_full_tested": len(a_full_ps),
            "trial_resolved_diff_test": trial_resolved_test,
        }

    pooled = _pool(rows) if rows else {"n_sessions": 0}
    per_corpus = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        per_corpus[dataset] = _pool([r for r in rows if r["dataset"] == dataset])

    rank_significant = bool(pooled.get("fractional_rank_test", {}).get("significant_two_sided", False)
                             and pooled.get("fractional_rank_test", {}).get("mean_value", 1.0) < 0.5)
    trial_resolved_significant_positive = bool(
        pooled.get("trial_resolved_diff_test", {}).get("significant", False)
        and pooled.get("trial_resolved_diff_test", {}).get("mean_value", -1.0) > 0.0)
    trial_resolved_significant_negative = bool(
        pooled.get("trial_resolved_diff_test", {}).get("status") == "tested"
        and pooled.get("trial_resolved_diff_test", {}).get("mean_value", 0.0) < 0.0
        and pooled.get("trial_resolved_diff_test", {}).get("significant_two_sided", False))
    a_minus1_survives = bool(rows and sum(1 for r in rows if r["subtractive"].get("a_minus1_clears_own_null")) >= len(rows) / 2)
    any_a_full_clears = bool(pooled.get("n_sessions_a_full_clears_own_null", 0) >= 1)
    rank_significant_reverse = bool(pooled.get("fractional_rank_test", {}).get("significant_two_sided", False)
                                     and pooled.get("fractional_rank_test", {}).get("mean_value", 0.0) > 0.5)

    if not rows or not any_a_full_clears:
        branch = "no_content_decodable"
    elif rank_significant_reverse:
        branch = "unattributed_off_branch_list_leading_latent_removal_improves_decoding"
    elif rank_significant and trial_resolved_significant_positive:
        branch = "content_in_the_state"
    elif rank_significant and a_minus1_survives:
        branch = "content_in_both"
    elif (not rank_significant) and not trial_resolved_significant_positive:
        branch = "content_beside_the_state"
    else:
        branch = "unattributed_off_branch_list"

    scope_by_corpus = {}
    for dataset in sorted({r["dataset"] for r in rows} | {r["dataset"] for r in status_rows}):
        tested = [r for r in rows if r["dataset"] == dataset]
        statuses = [r for r in status_rows if r["dataset"] == dataset]
        label_field = tested[0]["label_field"] if tested else (statuses[0].get("label_field") if statuses else None)
        scope_by_corpus[dataset] = {
            "label_field": label_field, "n_sessions_tested": len(tested),
            "n_sessions_status_row": len(statuses),
            "status_reasons": sorted({s.get("reason") or s.get("status") for s in statuses}),
            "n_classes_median": float(np.median([r["subtractive"]["n_classes"] for r in tested])) if tested else None,
            "chance_level": 0.5,
            "chance_level_note": "macro one-vs-rest AUC; chance is 0.5 regardless of class count",
        }

    output = {
        "version": "2026-08-15",
        "scope": (
            "Every (dataset, structure, session) reachable through src/corpus_sessions with a usable "
            "per-trial content label; delay epoch; 100 ms bins; window means over the whole delay epoch "
            "(a documented simplification of 'window means' plural -- one window spanning the epoch, not "
            "per-lag decoding -- for compute-budget reasons). k = min(8, n_units-2, n_trials//8) "
            "PCA latents, matching the project's existing content-CTG default of 8 where affordable. "
            "The trial-resolved test runs the margin split only (not a separate correct/incorrect split) "
            "and reports d_perm existence + early-segment slope, not the full late-segment/whole-range "
            "battery, per subset."
        ),
        "n_sessions_seen": n_sessions_seen, "n_sessions_tested": len(rows), "n_sessions_status_row": len(status_rows),
        "clearing_lags_bins_used": clearing_lags, "breakpoint_bins_used": breakpoint_bins,
        "deciding_branch": branch,
        "pooled": pooled, "per_corpus": per_corpus, "scope_by_corpus": scope_by_corpus,
        "session_rows": rows, "status_rows": status_rows,
        "wall_clock_s": time.time() - t0,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {OUTPUT_PATH} in {time.time()-t0:.1f}s", file=sys.stderr)
    print(json.dumps({"deciding_branch": branch, "n_sessions_tested": len(rows),
                       "fractional_rank_test": pooled.get("fractional_rank_test")}, indent=2))


if __name__ == "__main__":
    main()
