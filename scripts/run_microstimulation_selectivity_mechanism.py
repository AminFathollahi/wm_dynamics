#!/usr/bin/env python3
"""Does delay-period microstimulation raise firing (spikes/s) more on non-preferred targets than
on the unit's own preferred target, in the macaque dlPFC microstimulation corpus, under two
preferred-target selection windows, pooled per unit by multiple-imputation pooling across splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _sub in ("src", "scripts"):
    _p = str(ROOT / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from provenance import _json_safe, checkpoint_safe, git_commit, restore_checkpoint  # noqa: E402
from statistics import bootstrap_ci, minimum_detectable_paired_difference, paired_sign_flip_test, stable_seed  # noqa: E402
from selectivity_test import two_stage_selectivity_test  # noqa: E402
from io_utils import locked_json_update  # noqa: E402
from run_macaque_pfc_microstimulation_pipeline import DATA, SESSIONS, load_macaque_pfc_microstimulation_session  # noqa: E402
from run_macaque_pfc_microstimulation_design_corrected import BIN_MS, N_BINS, ONSET_BIN, bin_spiketrain  # noqa: E402

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints"
SCHEMA_VERSION = "microstimulation_selectivity_mechanism_v2"

RATE_UNIT = "spikes/s"
N_SPLITS = 5
ARTIFACT_BINS = 4
PRE_WINDOW = (0, 10)
POST_WINDOW = (ONSET_BIN + ARTIFACT_BINS, N_BINS)
MIN_GROUP_TRIALS = 2
SELECTIVITY_N_PERM = 2000
UNIT_SETS = ("all_units", "selective_units")
KINDS = ("preferred", "nonpreferred_pooled", "nonpreferred_anti", "difference_pooled", "difference_anti")
VARIANTS = ("raw", "corrected")
NONPREFERRED_DEFINITIONS = (
    ("pooled_remaining", "nonpreferred_pooled", "difference_pooled"),
    ("anti_preferred", "nonpreferred_anti", "difference_anti"),
)
# both readings select the preferred/anti-preferred target, and run the selectivity test, on
# CONTROL trials only (no stimulation confound); they differ only in which window that selection
# uses, matching it either to the pre-stimulation delay period or to the same window the
# post-stimulation rate change is measured in.
READINGS = {
    "preferred_from_pre_stimulation_window": "pre",
    "preferred_from_post_stimulation_control_window": "post",
}


def _load_session_arrays(prefix: str) -> dict | None:
    loaded = load_macaque_pfc_microstimulation_session(prefix, correct=True, neural_field="spiketrain")
    if loaded is None or loaded["control_idx"] is None:
        return None
    control_idx = loaded["control_idx"]
    pre_rows, post_rows, angle_rows, stim_rows = [], [], [], []
    for trial in loaded["trials"]:
        counts = bin_spiketrain(trial["spikerate"])
        if counts is None:
            continue
        pre_rows.append(counts[PRE_WINDOW[0]:PRE_WINDOW[1]].mean(axis=0) / (BIN_MS / 1000.0))
        post_rows.append(counts[POST_WINDOW[0]:POST_WINDOW[1]].mean(axis=0) / (BIN_MS / 1000.0))
        angle_rows.append(int(trial["angle_idx"]))
        stim_rows.append(trial["stim_cond"] != control_idx)
    if not pre_rows:
        return None
    return {
        "pre": np.array(pre_rows), "post": np.array(post_rows),
        "angle": np.array(angle_rows), "is_stim": np.array(stim_rows),
        "channel_ids": np.asarray(loaded["channel_ids"], dtype=int),
        "n_angle": int(max(angle_rows)) + 1,
    }


def _split_mask(angle: np.ndarray, is_stim: np.ndarray, seed_name: str) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed_name))
    train = np.zeros(len(angle), dtype=bool)
    for a in range(int(angle.max()) + 1):
        for flag in (False, True):
            idx = np.flatnonzero((angle == a) & (is_stim == flag))
            if len(idx) < 2:
                continue
            perm = rng.permutation(idx)
            train[perm[:len(perm) // 2]] = True
    return train


def _angle_held_out_effects(pre: np.ndarray, post: np.ndarray, angle: np.ndarray, is_stim: np.ndarray,
                            test: np.ndarray, n_angle: int) -> dict:
    out = {}
    for a in range(n_angle):
        control = test & ~is_stim & (angle == a)
        stim = test & is_stim & (angle == a)
        if control.sum() < MIN_GROUP_TRIALS or stim.sum() < MIN_GROUP_TRIALS:
            out[a] = None
            continue
        control_pre_mean = pre[control].mean(axis=0)
        control_post_mean = post[control].mean(axis=0)
        pre_diff = pre[stim] - control_pre_mean[None, :]
        raw_diff = post[stim] - control_post_mean[None, :]
        out[a] = {"raw": raw_diff, "corrected": raw_diff - pre_diff}
    return out


def _summarize(values: np.ndarray) -> tuple[float, float] | None:
    n = len(values)
    if n == 0:
        return None
    estimate = float(np.mean(values))
    variance = float(np.var(values, ddof=1) / n) if n >= 2 else 0.0
    return estimate, variance


def _pooled_nonpreferred(angle_effects: dict, pref_a: int, n_angle: int, variant: str, c: int) -> np.ndarray:
    parts = [angle_effects[a][variant][:, c] for a in range(n_angle)
            if a != pref_a and angle_effects.get(a) is not None]
    return np.concatenate(parts) if parts else np.array([])


def _pool_multiple_imputation(pairs: list[tuple[float, float]]) -> dict | None:
    m = len(pairs)
    if m == 0:
        return None
    estimates = np.array([p[0] for p in pairs])
    variances = np.array([p[1] for p in pairs])
    ubar = float(variances.mean())
    total = ubar + (1 + 1.0 / m) * float(np.var(estimates, ddof=1)) if m >= 2 else ubar
    return {"estimate": float(estimates.mean()), "variance": total, "m": m}


def _preferred_targets(selection: np.ndarray, train_control: np.ndarray, angle: np.ndarray, n_angle: int,
                       n_channels: int) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        means_by_angle = np.stack([
            selection[train_control & (angle == a)].mean(axis=0)
            if (train_control & (angle == a)).sum() else np.full(n_channels, np.nan)
            for a in range(n_angle)
        ])
        return np.nanargmax(means_by_angle, axis=0), np.nanargmin(means_by_angle, axis=0)


def _analyze_arrays(pre: np.ndarray, post: np.ndarray, angle: np.ndarray, is_stim: np.ndarray,
                    channel_ids: np.ndarray, n_angle: int, prefix: str) -> dict:
    n_channels = pre.shape[1]
    accum = {
        reading: {
            unit_set: {kind: {variant: [[] for _ in range(n_channels)] for variant in VARIANTS} for kind in KINDS}
            for unit_set in UNIT_SETS
        }
        for reading in READINGS
    }
    n_splits_usable = 0
    for split_idx in range(N_SPLITS):
        train = _split_mask(angle, is_stim, f"microstim_split_{prefix}_{split_idx}")
        test = ~train
        train_control = train & ~is_stim
        if train_control.sum() < n_angle * MIN_GROUP_TRIALS:
            continue
        n_splits_usable += 1

        angle_effects = _angle_held_out_effects(pre, post, angle, is_stim, test, n_angle)
        train_angle_labels = angle[train_control]
        train_trial_id = np.arange(train_control.sum())
        sources = {"pre": pre, "post": post}

        for reading, source_name in READINGS.items():
            selection = sources[source_name]
            pref_by_unit, antipref_by_unit = _preferred_targets(selection, train_control, angle, n_angle,
                                                                 n_channels)
            sel_rng = np.random.default_rng(stable_seed(f"microstim_selectivity_{reading}_{prefix}_{split_idx}"))
            for c in range(n_channels):
                sel = two_stage_selectivity_test(selection[train_control, c], train_angle_labels, train_trial_id,
                                                 sel_rng, n_perm=SELECTIVITY_N_PERM)
                unit_sets_for_channel = ("all_units", "selective_units") \
                    if sel.get("meets_selectivity_criterion") else ("all_units",)

                pref_a, antipref_a = int(pref_by_unit[c]), int(antipref_by_unit[c])
                pref_effect = angle_effects.get(pref_a)
                antipref_effect = angle_effects.get(antipref_a) if antipref_a != pref_a else None

                for variant in VARIANTS:
                    pref_s = _summarize(pref_effect[variant][:, c]) if pref_effect is not None else None
                    nonpref_s = _summarize(_pooled_nonpreferred(angle_effects, pref_a, n_angle, variant, c))
                    antipref_s = _summarize(antipref_effect[variant][:, c]) if antipref_effect is not None else None
                    diff_pooled = (nonpref_s[0] - pref_s[0], nonpref_s[1] + pref_s[1]) \
                        if pref_s is not None and nonpref_s is not None else None
                    diff_anti = (antipref_s[0] - pref_s[0], antipref_s[1] + pref_s[1]) \
                        if pref_s is not None and antipref_s is not None else None

                    for unit_set in unit_sets_for_channel:
                        bucket = accum[reading][unit_set]
                        if pref_s is not None:
                            bucket["preferred"][variant][c].append(pref_s)
                        if nonpref_s is not None:
                            bucket["nonpreferred_pooled"][variant][c].append(nonpref_s)
                        if antipref_s is not None:
                            bucket["nonpreferred_anti"][variant][c].append(antipref_s)
                        if diff_pooled is not None:
                            bucket["difference_pooled"][variant][c].append(diff_pooled)
                        if diff_anti is not None:
                            bucket["difference_anti"][variant][c].append(diff_anti)

    summary = {}
    for reading in READINGS:
        summary[reading] = {}
        for unit_set in UNIT_SETS:
            summary[reading][unit_set] = {}
            for kind in KINDS:
                summary[reading][unit_set][kind] = {}
                for variant in VARIANTS:
                    per_unit = []
                    for c in range(n_channels):
                        pooled = _pool_multiple_imputation(accum[reading][unit_set][kind][variant][c])
                        if pooled is not None:
                            per_unit.append({"channel_id": int(channel_ids[c]), **pooled})
                    summary[reading][unit_set][kind][variant] = {
                        "per_unit": per_unit, "n_units": len(per_unit),
                        "session_mean": float(np.mean([u["estimate"] for u in per_unit])) if per_unit else None,
                    }
    return {"n_channels": int(n_channels), "n_angle": int(n_angle), "n_splits_usable": n_splits_usable,
            "summary": summary}


def analyze_session(prefix: str) -> dict:
    arrays = _load_session_arrays(prefix)
    if arrays is None:
        return {"session": prefix, "status": "excluded", "reason": "no usable trials or no control condition"}
    result = _analyze_arrays(arrays["pre"], arrays["post"], arrays["angle"], arrays["is_stim"],
                             arrays["channel_ids"], arrays["n_angle"], prefix)
    if result["n_splits_usable"] == 0:
        return {"session": prefix, "status": "excluded", "reason": "no usable split"}
    return {"session": prefix, "animal": prefix[:2], "status": "complete", "n_trials": int(len(arrays["angle"])),
            **result}


def _checkpoint_path(prefix: str) -> Path:
    return CHECKPOINT_DIR / f"microstimulation_selectivity_mechanism_{SCHEMA_VERSION}_{prefix}.json"


def load_checkpoint(prefix: str) -> dict | None:
    path = _checkpoint_path(prefix)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != SCHEMA_VERSION:
        return None
    return restore_checkpoint(payload["result"])


def save_checkpoint(prefix: str, result: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": SCHEMA_VERSION, "result": checkpoint_safe(result)}
    _checkpoint_path(prefix).write_text(json.dumps(payload))


def _pool_across_sessions(sessions: list[dict], reading: str, unit_set: str, kind: str, variant: str) -> dict:
    values, n_units_list, animals = [], [], []
    for s in sessions:
        entry = s["summary"][reading][unit_set][kind][variant]
        if entry["n_units"] == 0:
            continue
        values.append(entry["session_mean"])
        n_units_list.append(entry["n_units"])
        animals.append(s["animal"])
    if len(values) < 2:
        return {"status": "not_computable", "n_sessions": len(values)}
    values_arr = np.array(values)
    tag = f"{reading}_{unit_set}_{kind}_{variant}"
    est_ci = bootstrap_ci(values_arr, lambda x: float(np.mean(x)),
                          rng=np.random.default_rng(stable_seed(f"microstim_boot_{tag}")))
    sign_flip = paired_sign_flip_test(values_arr, np.zeros_like(values_arr), alternative="two-sided",
                                      rng=np.random.default_rng(stable_seed(f"microstim_signflip_{tag}")))
    mdd = minimum_detectable_paired_difference(values_arr)
    per_animal = {}
    for animal in sorted(set(animals)):
        vals = np.array([v for v, a in zip(values, animals) if a == animal])
        entry = {"n_sessions": int(len(vals)), "mean": float(vals.mean())}
        if len(vals) >= 2:
            animal_ci = bootstrap_ci(vals, lambda x: float(np.mean(x)),
                                     rng=np.random.default_rng(stable_seed(f"microstim_animal_boot_{animal}_{tag}")))
            animal_sign = paired_sign_flip_test(vals, np.zeros_like(vals), alternative="two-sided",
                                                rng=np.random.default_rng(stable_seed(f"microstim_animal_signflip_{animal}_{tag}")))
            entry["ci"] = [animal_ci[1], animal_ci[2]]
            entry["p_value"] = animal_sign["p_value"]
        per_animal[animal] = entry
    return {"status": "computed", "n_sessions": len(values), "n_units": int(sum(n_units_list)),
            "estimate": est_ci[0], "ci": [est_ci[1], est_ci[2]], "p_value": sign_flip["p_value"],
            "minimum_detectable_difference": mdd, "per_animal": per_animal,
            "session_values": [float(v) for v in values]}


def _negate_pooled(pooled: dict) -> dict:
    if pooled.get("status") != "computed":
        return pooled
    negated = dict(pooled)
    negated["estimate"] = -pooled["estimate"]
    negated["ci"] = [-pooled["ci"][1], -pooled["ci"][0]]
    negated["session_values"] = [-v for v in pooled["session_values"]]
    negated["per_animal"] = {
        a: {**v, "mean": -v["mean"], **({"ci": [-v["ci"][1], -v["ci"][0]]} if "ci" in v else {})}
        for a, v in pooled["per_animal"].items()
    }
    return negated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=None)
    args = parser.parse_args()
    if "__WM_DYNAMICS_DATA_ROOT_NOT_SET__" in str(DATA) or not DATA.is_dir():
        raise SystemExit("Set WM_DYNAMICS_DATA_ROOT; macaque microstimulation data directory unavailable.")

    chosen = SESSIONS[:args.sessions] if args.sessions else SESSIONS
    per_session = []
    for prefix in chosen:
        cached = load_checkpoint(prefix)
        if cached is not None:
            print(f"checkpoint hit {prefix}", flush=True)
            per_session.append(cached)
            continue
        print(f"analyzing {prefix}", flush=True)
        result = analyze_session(prefix)
        save_checkpoint(prefix, result)
        per_session.append(result)

    complete = [s for s in per_session if s.get("status") == "complete"]

    pooled = {}
    for reading in READINGS:
        pooled[reading] = {}
        for unit_set in UNIT_SETS:
            pooled[reading][unit_set] = {}
            for definition, nonpref_kind, diff_kind in NONPREFERRED_DEFINITIONS:
                pooled[reading][unit_set][definition] = {}
                for variant in VARIANTS:
                    preferred = _pool_across_sessions(complete, reading, unit_set, "preferred", variant)
                    nonpreferred = _pool_across_sessions(complete, reading, unit_set, nonpref_kind, variant)
                    difference = _pool_across_sessions(complete, reading, unit_set, diff_kind, variant)
                    pooled[reading][unit_set][definition][variant] = {
                        "preferred_change": preferred,
                        "nonpreferred_change": nonpreferred,
                        "difference_nonpreferred_minus_preferred": difference,
                        "selectivity_index_change_preferred_minus_nonpreferred": _negate_pooled(difference),
                    }

    output = {
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(ROOT, __file__),
        "question": "does microstimulation raise firing more on non-preferred targets than on the "
            "unit's own preferred target",
        "rate_unit": RATE_UNIT,
        "pre_window_bins": list(PRE_WINDOW), "post_window_bins": list(POST_WINDOW),
        "onset_bin": ONSET_BIN, "artifact_excluded_bins": [ONSET_BIN, ONSET_BIN + ARTIFACT_BINS - 1],
        "bin_width_s": BIN_MS / 1000.0, "n_splits": N_SPLITS,
        "readings": {name: f"preferred and anti-preferred target chosen, and selectivity tested, on "
                    f"control trials' {source}-stimulation window" for name, source in READINGS.items()},
        "n_sessions_complete": len(complete), "n_sessions_excluded": len(per_session) - len(complete),
        "per_session": per_session,
        "pooled": pooled,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "microstimulation_selectivity_mechanism.json").write_text(json.dumps(_json_safe(output), indent=2))
    with locked_json_update(RESULTS / "all_statistics.json") as stats:
        stats["microstimulation_selectivity_mechanism"] = output
    print(json.dumps(_json_safe(pooled), indent=2))


if __name__ == "__main__":
    main()
