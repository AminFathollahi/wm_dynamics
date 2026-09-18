from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import _json_safe, git_commit  # noqa: E402
from run_human_maintenance_behaviour_link import (  # noqa: E402
    _session_trial_arrays, run_trial_admission_census,
)
from corpus_sessions import data_root  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "human_behavioral_value.json"
MODEL_NAMES = (
    "state", "task_load", "rate", "trial_history", "combined_nuisance", "combined_state",
)


def _history_features(session: dict) -> np.ndarray:
    y = np.asarray(session["is_correct"], dtype=float)
    prev = np.concatenate(([np.nan], y[:-1]))
    prev2 = np.concatenate(([np.nan, np.nan], y[:-2]))
    return np.column_stack((np.nan_to_num(prev, nan=0.0), np.nan_to_num(prev2, nan=0.0),
                            np.isfinite(prev).astype(float), np.isfinite(prev2).astype(float)))


def _records(sessions_by_corpus: dict[str, list[dict]]) -> list[dict]:
    records = []
    for corpus, sessions in sessions_by_corpus.items():
        for entry in sessions:
            if not entry.get("usable_for_estimator"):
                continue
            arrays = _session_trial_arrays(entry)
            if arrays.get("status") != "computed":
                continue
            hist = _history_features({"is_correct": arrays["is_correct"]})
            for i in range(len(arrays["is_correct"])):
                records.append({
                    "participant": f"{corpus}::{arrays['patient']}", "corpus": corpus,
                    "session": arrays["session"],
                    "y": float(arrays["is_correct"][i]), "state": float(arrays["deviation"][i]),
                    "load": int(arrays["load_level"][i]), "rate": float(arrays["spike_count"][i]),
                    "history": hist[i].tolist(),
                })
    return records


def _feature_matrix(rows: list[dict], name: str, train: bool, categories: dict | None = None):
    if name == "state":
        x = np.asarray([[r["state"]] for r in rows], dtype=float)
        return x, categories
    if name == "rate":
        return np.log1p(np.asarray([[max(r["rate"], 0.0)] for r in rows], dtype=float)), categories
    if name == "trial_history":
        return np.asarray([r["history"] for r in rows], dtype=float), categories
    loads = np.asarray([r["load"] for r in rows], dtype=int)
    corpora = np.asarray([r["corpus"] for r in rows], dtype=str)
    if train:
        categories = {"loads": np.unique(loads), "corpora": np.unique(corpora)}
    if categories is None or not len(categories["loads"]) or not len(categories["corpora"]):
        raise ValueError("load labels unavailable for requested task/load baseline")
    load_features = (loads[:, None] == categories["loads"][None, :]).astype(float)
    corpus_features = (corpora[:, None] == categories["corpora"][None, :]).astype(float)
    task_features = np.column_stack((corpus_features, load_features))
    if name == "task_load":
        return task_features, categories
    rate = np.log1p(np.asarray([[max(r["rate"], 0.0)] for r in rows], dtype=float))
    history = np.asarray([r["history"] for r in rows], dtype=float)
    nuisance = np.column_stack((task_features, rate, history))
    if name == "combined_nuisance":
        return nuisance, categories
    state = np.asarray([[r["state"]] for r in rows], dtype=float)
    return np.column_stack((nuisance, state)), categories


def _fit_predict(train_rows: list[dict], test_rows: list[dict], name: str) -> np.ndarray:
    if not train_rows or not test_rows:
        raise ValueError("empty participant-held-out split")
    y_train = np.asarray([r["y"] for r in train_rows], dtype=float)
    if np.unique(y_train).size < 2:
        raise ValueError("training split has no outcome variation")
    x_train, categories = _feature_matrix(train_rows, name, True)
    x_test, _ = _feature_matrix(test_rows, name, False, categories)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=0))
    model.fit(x_train, y_train)
    return np.clip(model.predict_proba(x_test)[:, 1], 1e-6, 1.0 - 1e-6)


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def _calibration(y: np.ndarray, p: np.ndarray) -> dict:
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(z, y)
    return {"brier": float(np.mean((p - y) ** 2)), "ece_10_bin": _ece(y, p),
            "calibration_intercept": float(model.intercept_[0]), "calibration_slope": float(model.coef_[0, 0])}


def _bootstrap_ci(values: np.ndarray, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)])
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def evaluate_records(records: list[dict]) -> dict:
    required = {"participant", "corpus", "y", "state", "load", "rate", "history"}
    if records and any(not required.issubset(row) for row in records):
        return {"status": "blocked", "reason": "released labels cannot support a requested baseline"}
    participants = sorted({r["participant"] for r in records})
    eligibility = {"n_rows": len(records), "n_participants": len(participants), "folds": {}}
    oof = {name: [] for name in MODEL_NAMES}
    fold_rows = {name: [] for name in MODEL_NAMES}
    excluded = {}
    for participant in participants:
        train = [r for r in records if r["participant"] != participant]
        test = [r for r in records if r["participant"] == participant]
        try:
            predictions = {}
            for name in MODEL_NAMES:
                predictions[name] = _fit_predict(train, test, name)
            y = np.asarray([r["y"] for r in test], dtype=float)
            for name, pred in predictions.items():
                oof[name].extend(zip(y.tolist(), pred.tolist()))
                fold_rows[name].append({"participant": participant, "y": y, "p": pred})
            eligibility["folds"][participant] = {"status": "computed", "n_test": len(test)}
        except (ValueError, KeyError) as exc:
            excluded[participant] = str(exc)
            eligibility["folds"][participant] = {"status": "excluded", "n_test": len(test), "reason": str(exc)}
    eligible = [p for p, row in eligibility["folds"].items() if row["status"] == "computed"]
    if not eligible:
        return {"status": "blocked", "reason": "no eligible participant-held-out folds", "eligibility": eligibility}
    metrics = {}
    for name in MODEL_NAMES:
        y = np.asarray([v[0] for v in oof[name]], dtype=float)
        p = np.asarray([v[1] for v in oof[name]], dtype=float)
        participant_losses = np.asarray([log_loss(row["y"], row["p"], labels=[0, 1]) for row in fold_rows[name]])
        metrics[name] = {"log_loss": float(log_loss(y, p, labels=[0, 1])), "n_rows": len(y),
                         "n_participants": len(participant_losses), "calibration": _calibration(y, p),
                         "participant_log_loss": {"mean": float(participant_losses.mean()),
                             "ci_lower": _bootstrap_ci(participant_losses, 17)[0], "ci_upper": _bootstrap_ci(participant_losses, 17)[1]}}
    comparisons = {}
    full_losses = {
        row["participant"]: log_loss(row["y"], row["p"], labels=[0, 1])
        for row in fold_rows["combined_state"]
    }
    for name in MODEL_NAMES:
        if name == "combined_state":
            continue
        baseline_losses = {row["participant"]: log_loss(row["y"], row["p"], labels=[0, 1]) for row in fold_rows[name]}
        diffs = np.asarray([baseline_losses[p] - full_losses[p] for p in eligible])
        lo, hi = _bootstrap_ci(diffs, 31 + list(MODEL_NAMES).index(name))
        comparisons[name] = {"estimand": "baseline_log_loss_minus_combined_state_log_loss", "participant_mean": float(diffs.mean()),
                             "ci_lower": lo, "ci_upper": hi, "n_participants": len(diffs),
                             "participant_differences": {p: float(d) for p, d in zip(eligible, diffs)}}
    return {"status": "complete", "eligibility": {**eligibility, "n_participants_eligible": len(eligible), "n_participants_excluded": len(excluded)},
            "metrics": metrics, "matched_prediction_differences": comparisons,
            "native_units": {"outcome": "trial correctness (0/1)", "log_loss": "natural log loss per trial", "calibration": "probability scale"}}


def main() -> None:
    census, sessions = run_trial_admission_census(data_root())
    usable = {}
    for corpus, entries in sessions.items():
        usable[corpus] = [a for a in entries if a.get("usable_for_estimator")]
    result = evaluate_records(_records(usable))
    result["trial_admission_census"] = census
    result["analysis_version"] = "2026-09-18"
    result["git_commit"] = git_commit(ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".partial")
    tmp.write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False))
    os.replace(tmp, OUTPUT_PATH)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT_PATH)}))


if __name__ == "__main__":
    main()
