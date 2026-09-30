from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from project_config import data_root
from provenance import _json_safe, git_commit  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from statistics import (  # noqa: E402
    Z_80_POWER, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed,
)
from stimulation_events import _censored_first_recall, _first_recall, contrast_summary, load_corpus  # noqa: E402

RESULTS = ROOT / "results"








def _subject_condition_values(lists: dict, extractor) -> tuple[list[float], list[float]]:
    stim, unstim = [], []
    for entry in lists.values():
        value = extractor(entry)
        if value is None:
            continue
        (stim if entry["stim_list"] else unstim).append(value)
    return stim, unstim




def _words_recalled(entry: dict) -> float:
    return float(entry["n_recalled"])


def _all_rec_events(entry: dict, field: str) -> list[float]:
    return entry[field]


def _is_zero_recall_list(entry: dict) -> float:
    return 0.0 if entry["rec_word_rt"] else 1.0




def censored_first_recall_contrast(by_subject_lists: dict, min_lists: int = 3) -> dict[str, dict]:
    from statsmodels.duration.survfunc import SurvfuncRight

    contrasts = {}
    for subject, lists in by_subject_lists.items():
        stim_pairs, unstim_pairs = [], []
        for entry in lists.values():
            pair = _censored_first_recall(entry)
            if pair is None:
                continue
            (stim_pairs if entry["stim_list"] else unstim_pairs).append(pair)
        if len(stim_pairs) < min_lists or len(unstim_pairs) < min_lists:
            continue
        stim_time, stim_status = (np.array(x) for x in zip(*stim_pairs))
        unstim_time, unstim_status = (np.array(x) for x in zip(*unstim_pairs))
        stim_median = SurvfuncRight(stim_time, stim_status).quantile(0.5)
        unstim_median = SurvfuncRight(unstim_time, unstim_status).quantile(0.5)
        if not (np.isfinite(stim_median) and np.isfinite(unstim_median)):
            continue
        contrasts[subject] = {"stim": float(stim_median), "unstim": float(unstim_median),
                              "n_stim_lists": len(stim_pairs), "n_unstim_lists": len(unstim_pairs),
                              "n_stim_events": int(stim_status.sum()), "n_unstim_events": int(unstim_status.sum())}
    return contrasts


def per_subject_contrast(by_subject_lists: dict, per_list_extractor, aggregate) -> dict[str, dict]:
    contrasts = {}
    for subject, lists in by_subject_lists.items():
        stim, unstim = _subject_condition_values(lists, per_list_extractor)
        if not stim or not unstim:
            continue
        contrasts[subject] = {"stim": aggregate(stim), "unstim": aggregate(unstim),
                              "n_stim_lists": len(stim), "n_unstim_lists": len(unstim)}
    return contrasts


def pooled_events_contrast(by_subject_lists: dict, field: str) -> dict[str, dict]:
    contrasts = {}
    for subject, lists in by_subject_lists.items():
        stim_events, unstim_events = [], []
        for entry in lists.values():
            (stim_events if entry["stim_list"] else unstim_events).extend(entry[field])
        if not stim_events or not unstim_events:
            continue
        contrasts[subject] = {"stim": float(np.median(stim_events)), "unstim": float(np.median(unstim_events)),
                              "n_stim_events": len(stim_events), "n_unstim_events": len(unstim_events)}
    return contrasts




def main() -> None:
    root = data_root()
    corpora = {"ram_ds005489_openloop": "ds005489-download", "ram_ds005557_closedloop": "ds005557-download"}
    per_corpus = {}
    pooled_by_metric: dict[str, list[dict]] = {
        "first_recall_latency_rec_word": [], "first_recall_latency_rec_word_plus_vv": [],
        "pooled_recall_latency_rec_word_plus_vv": [], "words_recalled_per_list": [],
        "fraction_zero_recall_lists": [], "censored_time_to_first_recall_rec_word": [],
    }

    for dataset, local_path in corpora.items():
        loaded = load_corpus(local_path, root)
        by_subject_lists = loaded["by_subject_lists"]

        first_recall_word = per_subject_contrast(
            by_subject_lists, lambda e: _first_recall(e, "rec_word_rt"), np.median)
        first_recall_all = per_subject_contrast(
            by_subject_lists, lambda e: _first_recall(e, "rec_all_rt"), np.median)
        pooled_recall_all = pooled_events_contrast(by_subject_lists, "rec_all_rt")
        words_recalled = per_subject_contrast(by_subject_lists, _words_recalled, np.mean)
        fraction_zero_recall = per_subject_contrast(by_subject_lists, _is_zero_recall_list, np.mean)
        censored_first_recall = censored_first_recall_contrast(by_subject_lists)

        for metric_key, contrasts in (
            ("first_recall_latency_rec_word", first_recall_word),
            ("first_recall_latency_rec_word_plus_vv", first_recall_all),
            ("words_recalled_per_list", words_recalled),
            ("fraction_zero_recall_lists", fraction_zero_recall),
            ("censored_time_to_first_recall_rec_word", censored_first_recall),
        ):
            pooled_by_metric[metric_key].extend(
                [{"subject": f"{dataset}:{s}", **c} for s, c in contrasts.items()])
        pooled_by_metric["pooled_recall_latency_rec_word_plus_vv"].extend(
            [{"subject": f"{dataset}:{s}", "stim": c["stim"], "unstim": c["unstim"]}
             for s, c in pooled_recall_all.items()])

        per_corpus[dataset] = {
            "n_sessions_seen": loaded["n_sessions"], "n_events_seen": loaded["n_events"],
            "n_subjects_seen": len(by_subject_lists),
            "first_recall_latency_rec_word": contrast_summary(first_recall_word, f"{dataset}_first_recall_word"),
            "first_recall_latency_rec_word_plus_vv": contrast_summary(first_recall_all, f"{dataset}_first_recall_all"),
            "pooled_recall_latency_rec_word_plus_vv": contrast_summary(pooled_recall_all, f"{dataset}_pooled_recall"),
            "words_recalled_per_list": contrast_summary(words_recalled, f"{dataset}_words_recalled"),
            "fraction_zero_recall_lists": contrast_summary(fraction_zero_recall, f"{dataset}_fraction_zero_recall"),
            "censored_time_to_first_recall_rec_word": contrast_summary(
                censored_first_recall, f"{dataset}_censored_first_recall"),
            "stimulation_parameter_inventory": {
                field: dict(counts) for field, counts in loaded["param_counts"].items()
            },
        }

    def _pool(metric_key: str, seed_name: str) -> dict:
        rows = pooled_by_metric[metric_key]
        contrasts = {row["subject"]: row for row in rows}
        return contrast_summary(contrasts, seed_name)

    pooled = {
        "first_recall_latency_rec_word": _pool("first_recall_latency_rec_word", "ram_pooled_first_recall_word"),
        "first_recall_latency_rec_word_plus_vv": _pool("first_recall_latency_rec_word_plus_vv", "ram_pooled_first_recall_all"),
        "pooled_recall_latency_rec_word_plus_vv": _pool("pooled_recall_latency_rec_word_plus_vv", "ram_pooled_pooled_recall"),
        "words_recalled_per_list": _pool("words_recalled_per_list", "ram_pooled_words_recalled"),
        "fraction_zero_recall_lists": _pool("fraction_zero_recall_lists", "ram_pooled_fraction_zero_recall"),
        "censored_time_to_first_recall_rec_word": _pool(
            "censored_time_to_first_recall_rec_word", "ram_pooled_censored_first_recall"),
        "note": "ram_ds005489_openloop and ram_ds005557_closedloop are two independent OpenNeuro releases "
                "with no declared patient overlap (config/datasets.json); pooling concatenates each "
                "corpus's own subject-level contrasts, subject identity resolved to the dataset directory.",
    }

    output = {
        "construct": "Three response-latency constructs are reported, plus a recall-count construct and "
            "a censored time-to-first-recall construct, and none are pooled into one number. "
            "first_recall_latency_rec_word (primary): within each (session, list) that has at least one "
            "REC_WORD event, the minimum response_time among REC_WORD events only -- always output "
            "position 1. This arm conditions on the list having at least one recall; lists with zero "
            "recalls carry no value here and are dropped from the per-subject median, so a condition that "
            "changes how often a list has zero recalls changes which lists this arm is computed over. "
            "fraction_zero_recall_lists (reported beside every latency arm) and "
            "censored_time_to_first_recall_rec_word (below) are the two constructs that keep zero-recall "
            "lists in view. first_recall_latency_rec_word_plus_vv: the same minimum-per-list definition "
            "but pooling REC_WORD with REC_WORD_VV -- kept as a sensitivity arm because a non-word "
            "vocalization is not a recalled list word; it conditions on the list having at least one "
            "REC_WORD or REC_WORD_VV event, the same way. pooled_recall_latency_rec_word_plus_vv "
            "(secondary, kept unchanged from the first delivery): the median response_time over every "
            "REC_WORD/REC_WORD_VV event in a condition's lists, pooled across output positions -- this "
            "quantity is what a reader computing directly from the released events tables would get, but "
            "it is entangled with recall count: response_time is cumulative within the recall period, so "
            "fewer recalled words truncates the slow late-output-position tail and can move this estimator "
            "as a side effect of an accuracy effect rather than a latency effect. It conditions on the "
            "condition's pooled list set having at least one recall event somewhere, not on any single "
            "list. words_recalled_per_list is reported alongside so that entanglement is visible rather "
            "than inferred: the mean count of WORD events with recalled=1 per list, stimulated minus "
            "unstimulated; it conditions on nothing, every list contributes a value including zero. "
            "fraction_zero_recall_lists: the mean, per subject and condition, of an indicator that a "
            "list has zero REC_WORD events; also conditions on nothing. "
            "censored_time_to_first_recall_rec_word: a list's time to first REC_WORD response_time, "
            "censored at the end of that list's own recall period (its REC_END minus REC_START onset) "
            "when the list has zero REC_WORD events; per subject and condition, the Kaplan-Meier median "
            "of these censored times (statsmodels.duration.survfunc.SurvfuncRight), then the "
            "stimulated-minus-unstimulated difference is a paired sign-flip test across subjects; this "
            "arm conditions on nothing being dropped, only on the recall period's own end time being an "
            "upper bound on an unobserved recall. None of these six numbers is a probe-locked recognition "
            "latency (~1.1-1.3 s median in other corpora in this project); this is a verbal free-recall "
            "latency, several seconds in scale.",
        "git_commit": git_commit(ROOT),
        "per_corpus": per_corpus,
        "pooled": pooled,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "ram_stimulation_response_latency.json").write_text(json.dumps(_json_safe(output), indent=2))
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["ram_stimulation_response_latency"] = output
    print(json.dumps({k: {"n_subjects_seen": v["n_subjects_seen"],
                           "first_recall_word_diff": v["first_recall_latency_rec_word"].get("mean_diff"),
                           "first_recall_word_p": v["first_recall_latency_rec_word"].get("p_value"),
                           "words_recalled_diff": v["words_recalled_per_list"].get("mean_diff"),
                           "fraction_zero_recall_stim": v["fraction_zero_recall_lists"].get("mean_stim"),
                           "fraction_zero_recall_unstim": v["fraction_zero_recall_lists"].get("mean_unstim"),
                           "censored_first_recall_diff": v["censored_time_to_first_recall_rec_word"].get("mean_diff"),
                           "censored_first_recall_p": v["censored_time_to_first_recall_rec_word"].get("p_value")}
                       for k, v in per_corpus.items()}, indent=2))
    print(json.dumps({"pooled_first_recall_word": pooled["first_recall_latency_rec_word"],
                       "pooled_words_recalled": pooled["words_recalled_per_list"],
                       "pooled_fraction_zero_recall": pooled["fraction_zero_recall_lists"],
                       "pooled_censored_first_recall": pooled["censored_time_to_first_recall_rec_word"]}, indent=2))


if __name__ == "__main__":
    main()
