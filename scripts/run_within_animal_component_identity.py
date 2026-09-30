from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for part in ("src", "scripts"):
    path = str(ROOT / part)
    if path not in sys.path:
        sys.path.insert(0, path)

from corpus_sessions import (  # noqa: E402
    data_root, independent_unit, iter_pfc4, load_watters_session, watters_behaviour, watters_session_dates,
)
from provenance import _json_safe, git_commit  # noqa: E402
from info_decoding import MAX_SESSIONS_ENV_VAR
from statistics import Z_80_POWER
from run_rank_free_component_identity import _standard_sessions
from info_decoding import _combine_levels, _finite
from statistics import fdr_bh, permutation_pvalue, stable_seed  # noqa: E402
from subspace_identity import block_folds, class_basis, permutation_alignment, regression_basis  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "within_animal_component_identity.json"
CHECKPOINT_DIR = ROOT / "results" / ".checkpoints" / "run_within_animal_component_identity"
SCHEMA_VERSION = "1.1"
VERSION = "2026-09-24"
SEED_NAMESPACE = f"within_animal_component_identity|{VERSION}"
N_FOLDS = 2
N_PERM = 1000
N_BOOT = 2000
MIN_TRIALS = 2 * max(3, N_FOLDS)
MIN_SESSIONS_PER_ANIMAL = 4
FDR_ALPHA = 0.05
REPRODUCTION_TOLERANCE = 1e-9
PFC4_CORPUS = "pfc4_macaque_somatosensory"
CORPORA = ("panichello_2024_macaque_lPFC", "watters_2026_macaque_multi_object", PFC4_CORPUS)
CANDIDATE_KEYS = ("gain_total_spike_count", "memorandum_content")

SCOPE = (
    "within-animal estimate: the same equal-independent-unit mean held-out squared projection of an "
    "evaluation-fold residual axis onto a training-fold label subspace as "
    "results/rank_free_component_identity.json, recomputed separately inside each animal with SESSION, "
    "not animal, as the resampling and clustering unit. This is a deliberately weaker, differently "
    "scoped estimand asked in the same data because all three macaque corpora carry too few animals (3, "
    "2 and 2) to ever clear the four-independent-unit floor the delivered cross-animal estimand requires "
    "-- no future release from any of the three laboratories can reach four animals, so that floor is "
    "not a sample-size problem this analysis, or any amount of new recording, can fix. Results are reported "
    "per animal, separately, and are NEVER pooled across animals into one number: no cross-animal "
    "generalisation claim is supported here. Clearing this analysis's own decision rule never counts as "
    "clearing, relaxing, meeting or substituting for the four-independent-unit floor -- it is a "
    "different, weaker question asked in the same data. A within-animal result here and the delivered "
    "cross-animal result in rank_free_component_identity.json are not replications of each other in "
    "either direction; neither is voided nor confirmed by the other."
)

DECISION_RULE = (
    "Pre-declared before any cell was fit, not amended afterwards. A candidate cell (one animal, one "
    "corpus, one of gain_total_spike_count or memorandum_content, pooled with SESSION as the resampling "
    "and clustering unit) CLEARS if and only if its permutation p-value survives Benjamini-Hochberg FDR "
    "at alpha=0.05 across every computed primary cell in this artifact AND its whole-session cluster "
    "bootstrap 95% interval for the mean alignment-above-null excludes zero. A cell clearing exactly one "
    "of the two conditions clears NEITHER, and both numbers (the FDR-surviving p/q and the bootstrap "
    "interval) are reported together, never singly. A cell that does not clear is converted to a bound "
    "where possible: its own minimum detectable difference at 80 percent power (z="
    f"{Z_80_POWER} times the whole-session-cluster-bootstrap standard error of its own mean "
    "alignment-above-null, never a trial-count formula or an intraclass-correlation design effect) is "
    "compared against that SAME animal's own gain_total_spike_count mean alignment-above-null as an "
    "internal, same-scale, same-corpus reference. If the minimum detectable difference is smaller in "
    "magnitude than that reference, the cell is a BOUNDED NEGATIVE: an effect of that magnitude would "
    "have been detected here and was not, while smaller effects are not excluded -- both halves of that "
    "sentence always travel together. If the minimum detectable difference is at or above the "
    "reference, the cell is INCONCLUSIVE. gain_total_spike_count is the reference candidate itself and "
    "is never bounded against itself: when it does not clear it is reported "
    "NOT_CLEARED_NO_INTERNAL_REFERENCE with its own minimum detectable difference and no bound "
    "classification. An animal with fewer than "
    f"{MIN_SESSIONS_PER_ANIMAL} sessions carrying a computed per-session cell for a candidate is "
    "NOT_COMPUTABLE for that candidate, reported with its own session count, never treated as a null. "
    "Infeasibility is never a null. memorandum_content_many_class_diagnostic (macaque multi-object corpus only: the same "
    "content candidate built as a many-class subspace from the corpus's own native theta values instead "
    "of the two-column sine/cosine axis) is a primary cell in this rule's FDR family, reported beside "
    "the two-column construction with the same fields, including verdict and bound."
)


def _hash() -> str:
    digest = hashlib.sha256()
    paths = (
        Path(__file__), ROOT / "src" / "subspace_identity.py", ROOT / "src" / "corpus_sessions.py",
        ROOT / "src" / "statistics.py", ROOT / "src" / "info_decoding.py", ROOT / "scripts" / "run_component_identity_subspace_atlas.py",
        ROOT / "scripts" / "run_rank_free_component_identity.py",
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _seed(seed_id: str) -> int:
    return stable_seed(f"{SEED_NAMESPACE}|{seed_id}")


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_json_safe(value), handle, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _checkpoint(key: str, identity: dict, fit) -> dict:
    readable = "".join(character if character.isalnum() or character in "-_" else "_" for character in key)[:80]
    suffix = hashlib.sha256(key.encode()).hexdigest()[:16]
    path = CHECKPOINT_DIR / f"{readable}.{suffix}.json"
    if path.exists():
        try:
            cached = json.loads(path.read_text())
            record = cached.get("record")
            if cached.get("identity") == identity and cached.get("complete") is True and isinstance(record, dict):
                return record
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            pass
    record = fit()
    _write(path, {"complete": True, "identity": identity, "record": record})
    return record


def _fit_cell(activity: np.ndarray, target: np.ndarray, kind: str, n_perm: int, seed: str) -> dict:
    activity = np.asarray(activity, dtype=float)
    norms = np.linalg.norm(activity, axis=1)
    valid = np.isfinite(activity).all(axis=1) & (norms > 0) & _finite(target)
    directions = activity[valid] / norms[valid, None]
    target = np.asarray(target)[valid]
    if len(target) < MIN_TRIALS:
        return {"status": "not_computable", "reason": "too few finite trials", "n_trials": int(len(target))}
    fit_basis = class_basis if kind == "categorical" else regression_basis
    result = permutation_alignment(
        directions, target, block_folds(len(target), N_FOLDS), fit_basis, n_perm,
        np.random.default_rng(_seed(seed)),
    )
    result["n_trials"] = len(target)
    result["target_kind"] = kind
    result["seed_id"] = seed
    return result


def _panichello_sessions(root: Path, n_perm: int, identity: dict) -> list[dict]:
    corpus = "panichello_2024_macaque_lPFC"
    records = []
    for session, reason, activity, categorical, continuous in _standard_sessions(root, corpus):
        animal = independent_unit(corpus, session)
        key = f"{corpus}|{session}"

        def fit(session=session, reason=reason, activity=activity, categorical=categorical,
                 continuous=continuous, animal=animal, key=key):
            if activity is None:
                return {"session": session, "independent_unit": animal, "status": "not_computable", "reason": reason}
            gain = _fit_cell(
                activity, continuous["gain_total_spike_count"], "continuous", n_perm, f"{key}|gain_total_spike_count",
            )
            content = _fit_cell(
                activity, categorical["memorandum_content"], "categorical", n_perm, f"{key}|memorandum_content",
            )
            return {
                "session": session, "independent_unit": animal, "status": "computed",
                "candidates": {"gain_total_spike_count": gain, "memorandum_content": content},
            }

        records.append(_checkpoint(key, identity, fit))
    return records


def _watters_sessions(root: Path, n_perm: int, max_sessions: int | None, identity: dict) -> list[dict]:
    corpus = "watters_2026_macaque_multi_object"
    behaviour = watters_behaviour(root)
    dates = watters_session_dates(root)
    if max_sessions:
        dates = dates[:max_sessions]
    records = []
    for animal, date, variant in dates:
        session_id = f"{animal}_{date}_{variant}"
        key = f"{corpus}|{session_id}"

        def fit(animal=animal, date=date, session_id=session_id, key=key):
            session = load_watters_session(root, animal, date, behaviour)
            if session.get("status") != "loaded":
                return {
                    "session": session_id, "independent_unit": animal, "status": "not_computable",
                    "reason": session.get("status"),
                }
            counts = session["counts"]
            levels = session["num_objects"]
            theta = np.asarray(session["cued_theta"], dtype=float)
            cued = np.column_stack([np.cos(theta), np.sin(theta)])
            many_class_labels = np.round(theta, 6)
            gain_levels, content_levels, many_class_levels = [], [], []
            attempted = False
            for level in np.unique(levels):
                mask = levels == level
                if int(mask.sum()) < MIN_TRIALS:
                    continue
                attempted = True
                activity = counts[mask].sum(axis=2)
                gain_cell = _fit_cell(
                    activity, activity.sum(axis=1), "continuous", n_perm,
                    f"{key}|level={level}|gain_total_spike_count",
                )
                content_cell = _fit_cell(
                    activity, cued[mask], "continuous", n_perm, f"{key}|level={level}|memorandum_content",
                )
                many_cell = _fit_cell(
                    activity, many_class_labels[mask], "categorical", n_perm,
                    f"{key}|level={level}|memorandum_content_many_class",
                )
                gain_levels.append((gain_cell.get("n_trials", 0), gain_cell))
                content_levels.append((content_cell.get("n_trials", 0), content_cell))
                many_class_levels.append((many_cell.get("n_trials", 0), many_cell))
            if not attempted:
                return {
                    "session": session_id, "independent_unit": animal, "status": "not_computable",
                    "reason": "no item-count level reached the trial floor",
                }
            return {
                "session": session_id, "independent_unit": animal, "status": "computed",
                "candidates": {
                    "gain_total_spike_count": _combine_levels(gain_levels),
                    "memorandum_content": _combine_levels(content_levels),
                    "memorandum_content_many_class_diagnostic": _combine_levels(many_class_levels),
                },
            }

        records.append(_checkpoint(key, identity, fit))
    return records


def _pfc4_sessions(root: Path, n_perm: int, identity: dict) -> list[dict]:
    corpus = PFC4_CORPUS
    records = []
    for entry in iter_pfc4(root):
        animal, session = entry["patient"], entry["session"]
        key = f"{corpus}|{session}"

        def fit(entry=entry, animal=animal, session=session, key=key):
            activity = np.asarray(entry["counts"]).sum(axis=2)
            gain = _fit_cell(
                activity, activity.sum(axis=1), "continuous", n_perm, f"{key}|gain_total_spike_count",
            )
            content = _fit_cell(
                activity, entry["f1_hz"], "continuous", n_perm, f"{key}|memorandum_content",
            )
            return {
                "session": session, "independent_unit": animal, "status": "computed",
                "candidates": {"gain_total_spike_count": gain, "memorandum_content": content},
            }

        records.append(_checkpoint(key, identity, fit))
    return records


def _group_by_animal(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["independent_unit"], []).append(record)
    return grouped


def _pool_within_animal(cells: list[dict | None], seed: str) -> dict:
    computed = [cell for cell in cells if cell is not None and cell.get("status") == "computed"]
    n_sessions = len(computed)
    if n_sessions < MIN_SESSIONS_PER_ANIMAL:
        return {
            "status": "not_computable", "n_sessions": n_sessions,
            "reason": f"fewer than {MIN_SESSIONS_PER_ANIMAL} sessions carry a computed cell",
        }
    n_draws = min(len(cell["null_draws"]) for cell in computed)
    session_null_means = [float(np.asarray(cell["null_draws"])[:n_draws].mean()) for cell in computed]
    effects = np.asarray(
        [cell["alignment"] - null_mean for cell, null_mean in zip(computed, session_null_means)],
    )
    observed = float(np.mean([cell["alignment"] for cell in computed]))
    draws = np.mean(np.stack([np.asarray(cell["null_draws"])[:n_draws] for cell in computed]), axis=0)
    rng = np.random.default_rng(_seed(seed))
    boot = np.asarray([effects[rng.integers(0, len(effects), len(effects))].mean() for _ in range(N_BOOT)])
    se = float(np.std(boot, ddof=1))
    interval = [float(value) for value in np.percentile(boot, [2.5, 97.5])]
    if interval[0] > 0:
        interval_position = "interval_above_zero"
    elif interval[1] < 0:
        interval_position = "interval_below_zero"
    else:
        interval_position = "interval_spans_zero"
    null_center = float(draws.mean())
    return {
        "status": "computed",
        "n_sessions": n_sessions,
        "session_weighting": "equal",
        "mean_alignment": observed,
        "mean_null_alignment": null_center,
        "mean_alignment_above_null": float(effects.mean()),
        "session_cluster_bootstrap_interval_95pct": interval,
        "session_cluster_bootstrap_se": se,
        "minimum_detectable_difference_80pct_power": float(Z_80_POWER * se),
        "p_value": permutation_pvalue(draws >= observed),
        "p_value_two_sided": permutation_pvalue(np.abs(draws - null_center) >= abs(observed - null_center)),
        "n_permutations": int(len(draws)),
        "bootstrap_seed_id": seed,
        "signed_reading": {
            "estimate": float(effects.mean()),
            "interval_95pct": interval,
            "interval_position": interval_position,
        },
    }


CONTENT_CONSTRUCTION_NOTE = {
    "panichello_2024_macaque_lPFC": (
        "native categorical class-mean subspace over cueAngIdx, the corpus's own 8 discrete memorandum "
        "locations (confirmed 8 distinct cueAng values in the raw session files); subspace dimension is "
        "min(7, n_units), kept unchanged from the class-based construction because the memorandum is "
        "genuinely categorical here, not artificially discretised"
    ),
    "watters_2026_macaque_multi_object": (
        "circular-continuous axis: regression_basis fit on [cos(theta), sin(theta)] of the memorandum "
        "angle, a two-column continuous target giving a subspace dimension of min(2, n_units), in place "
        "of a many-class construction because the memorandum angle here is genuinely continuous"
    ),
    PFC4_CORPUS: (
        "one-dimensional continuous axis: regression_basis fit directly on the trial's F1 value in Hz "
        "(no sine/cosine pair, unlike the multi-object corpus angle, because frequency does not wrap), giving a "
        "subspace dimension of min(1, n_units). The raw session files carry 19 distinct F1 values (6 to "
        "48 Hz, mostly 2 Hz steps), wider than the 6 values the deposited data description names, so the "
        "6-value description understates the corpus's own range. Chosen over a class-based construction "
        "because the task defines F1 as a magnitude the animal holds across the delay and compares "
        "against F2 as higher-or-lower to produce the trial's hit/miss outcome -- an ordinal, interval "
        "quantity, not an arbitrary class identity like the spatial working-memory corpus's discrete spatial locations"
    ),
}


def _pool_corpus(corpus: str, records: list[dict]) -> dict:
    grouped = _group_by_animal(records)
    candidates = {}
    for candidate in CANDIDATE_KEYS:
        per_animal = {}
        for animal, animal_records in grouped.items():
            cells = [record.get("candidates", {}).get(candidate) for record in animal_records]
            pooled = _pool_within_animal(cells, f"{corpus}|{animal}|{candidate}|bootstrap")
            pooled["n_sessions_in_corpus"] = len(animal_records)
            per_animal[animal] = pooled
        candidates[candidate] = per_animal
    candidates["memorandum_content"] = {
        animal: {**cell, "content_construction": CONTENT_CONSTRUCTION_NOTE[corpus]}
        for animal, cell in candidates["memorandum_content"].items()
    }
    if corpus == "watters_2026_macaque_multi_object":
        per_animal = {}
        for animal, animal_records in grouped.items():
            cells = [
                record.get("candidates", {}).get("memorandum_content_many_class_diagnostic")
                for record in animal_records
            ]
            pooled = _pool_within_animal(cells, f"{corpus}|{animal}|memorandum_content_many_class_diagnostic|bootstrap")
            pooled["n_sessions_in_corpus"] = len(animal_records)
            pooled["content_construction"] = (
                "many-class subspace over the corpus's own native theta values (each distinct cued "
                "angle, rounded to 1e-6 radians, is its own class), the construction the one-dimensional "
                "circular-continuous axis above replaces"
            )
            per_animal[animal] = pooled
        candidates["memorandum_content_many_class_diagnostic"] = per_animal
    return candidates


PRIMARY_CANDIDATE_KEYS = (*CANDIDATE_KEYS, "memorandum_content_many_class_diagnostic")


def _finalize(per_corpus: dict[str, dict]) -> dict[str, dict]:
    primary = []
    for corpus, candidates in per_corpus.items():
        for candidate in PRIMARY_CANDIDATE_KEYS:
            for animal, cell in candidates.get(candidate, {}).items():
                if cell.get("status") == "computed":
                    primary.append((corpus, candidate, animal, cell))
    if primary:
        correction = fdr_bh(np.asarray([cell["p_value"] for *_, cell in primary]), alpha=FDR_ALPHA)
        for (_, _, _, cell), reject, q_value in zip(primary, correction["reject"], correction["q_values"]):
            cell["fdr_q_value"] = float(q_value)
            cell["fdr_reject"] = bool(reject)
            ci = cell["session_cluster_bootstrap_interval_95pct"]
            ci_excludes_zero = ci[0] > 0 or ci[1] < 0
            if reject and ci_excludes_zero:
                cell["verdict"] = "cleared"
            elif reject or ci_excludes_zero:
                cell["verdict"] = "clears_neither"
            else:
                cell["verdict"] = "not_cleared"
    for corpus, candidates in per_corpus.items():
        gain_by_animal = candidates["gain_total_spike_count"]
        for candidate in PRIMARY_CANDIDATE_KEYS:
            for animal, cell in candidates.get(candidate, {}).items():
                if cell.get("status") != "computed" or cell.get("verdict") == "cleared":
                    continue
                if candidate == "gain_total_spike_count":
                    cell["verdict"] = "not_cleared_no_internal_reference"
                    continue
                gain_cell = gain_by_animal.get(animal, {})
                if gain_cell.get("status") != "computed":
                    cell["bound"] = "inconclusive_no_internal_reference"
                    continue
                reference = abs(gain_cell["mean_alignment_above_null"])
                mdd = cell["minimum_detectable_difference_80pct_power"]
                cell["internal_reference_gain_effect"] = reference
                cell["bound"] = "bounded_negative" if mdd < reference else "inconclusive"
    return per_corpus


def _attach_many_class_comparison(per_corpus: dict[str, dict]) -> None:
    watters = per_corpus.get("watters_2026_macaque_multi_object")
    if not watters:
        return
    continuous = watters["memorandum_content"]
    many_class = watters.get("memorandum_content_many_class_diagnostic", {})
    for animal, cell in many_class.items():
        reference = continuous.get(animal, {})
        if cell.get("status") == "computed" and reference.get("status") == "computed":
            cell["continuous_axis_minimum_detectable_difference_80pct_power"] = (
                reference["minimum_detectable_difference_80pct_power"]
            )
            cell["many_class_minus_continuous_axis_minimum_detectable_difference"] = (
                cell["minimum_detectable_difference_80pct_power"]
                - reference["minimum_detectable_difference_80pct_power"]
            )


def _run_pipeline(
    root: Path, n_perm: int, max_sessions: int | None, checkpoint_dir: Path, identity: dict,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    global CHECKPOINT_DIR
    previous = CHECKPOINT_DIR
    CHECKPOINT_DIR = checkpoint_dir
    try:
        panichello_records = _panichello_sessions(root, n_perm, identity)
        watters_records = _watters_sessions(root, n_perm, max_sessions, identity)
        pfc4_records = _pfc4_sessions(root, n_perm, identity)
    finally:
        CHECKPOINT_DIR = previous
    per_corpus = {
        "panichello_2024_macaque_lPFC": _pool_corpus("panichello_2024_macaque_lPFC", panichello_records),
        "watters_2026_macaque_multi_object": _pool_corpus(
            "watters_2026_macaque_multi_object", watters_records,
        ),
        PFC4_CORPUS: _pool_corpus(PFC4_CORPUS, pfc4_records),
    }
    per_corpus = _finalize(per_corpus)
    _attach_many_class_comparison(per_corpus)
    return panichello_records, watters_records, pfc4_records, per_corpus


def _compare(a, b, path: str, differences: list[str], max_diff: list[float]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            differences.append(f"{path}: key mismatch {sorted(set(a) ^ set(b))}")
            return
        for key in a:
            _compare(a[key], b[key], f"{path}.{key}", differences, max_diff)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            differences.append(f"{path}: length mismatch {len(a)} vs {len(b)}")
            return
        for index, (av, bv) in enumerate(zip(a, b)):
            _compare(av, bv, f"{path}[{index}]", differences, max_diff)
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        diff = abs(float(a) - float(b))
        max_diff[0] = max(max_diff[0], diff)
        if diff > REPRODUCTION_TOLERANCE:
            differences.append(f"{path}: {a} vs {b} (abs diff {diff})")
        return
    if a != b:
        differences.append(f"{path}: {a!r} vs {b!r}")


def _reproduction_gate(root: Path, n_perm: int, max_sessions: int | None, identity: dict, reference: dict) -> dict:
    temporary = Path(tempfile.mkdtemp(prefix="within_animal_component_identity_reproduction_"))
    try:
        _, _, _, recomputed = _run_pipeline(root, n_perm, max_sessions, temporary, identity)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    differences: list[str] = []
    max_diff = [0.0]
    _compare(_json_safe(reference), _json_safe(recomputed), "summary", differences, max_diff)
    return {
        "status": "reproduced_exactly" if not differences else "reproduction_mismatch",
        "tolerance": REPRODUCTION_TOLERANCE,
        "max_absolute_difference": max_diff[0],
        "differences": differences[:50],
        "n_differences": len(differences),
    }


def main() -> None:
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--max-sessions", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    args = parser.parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    if args.n_perm < 1:
        parser.error("--n-perm must be positive")
    if args.max_sessions is not None and args.max_sessions < 1:
        parser.error("--max-sessions must be positive")
    if args.max_sessions is not None:
        os.environ[MAX_SESSIONS_ENV_VAR] = str(args.max_sessions)
    started = time.time()
    root = data_root()
    identity = {
        "version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "code_hash": _hash(),
        "data_root": str(root.resolve()),
        "n_perm": args.n_perm,
        "n_folds": N_FOLDS,
        "seed_namespace": SEED_NAMESPACE,
        "seed_algorithm": "statistics.stable_seed_crc32",
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "method": "cross_fitted_squared_subspace_alignment_within_animal",
        "scope": SCOPE,
        "decision_rule": DECISION_RULE,
        "identity": identity,
        "data_root": str(root),
        "corpora": {},
    }
    _write(args.output, output)

    panichello_records, watters_records, pfc4_records, per_corpus = _run_pipeline(
        root, args.n_perm, args.max_sessions, args.checkpoint_dir, identity,
    )
    output["corpora"]["panichello_2024_macaque_lPFC"] = {
        "records": [_strip(record) for record in panichello_records],
    }
    output["corpora"]["watters_2026_macaque_multi_object"] = {
        "records": [_strip(record) for record in watters_records],
    }
    output["corpora"][PFC4_CORPUS] = {
        "records": [_strip(record) for record in pfc4_records],
    }
    output["summary"] = {"per_corpus": per_corpus}
    _write(args.output, output)

    output["reproduction_gate"] = _reproduction_gate(
        root, args.n_perm, args.max_sessions, identity, output["summary"]["per_corpus"],
    )
    output["status"] = "complete"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(args.output, output)


def _strip(record: dict) -> dict:
    clean = dict(record)
    candidates = clean.get("candidates")
    if isinstance(candidates, dict):
        clean["candidates"] = {
            name: {key: value for key, value in cell.items() if key != "null_draws"}
            for name, cell in candidates.items()
        }
    return clean


if __name__ == "__main__":
    main()
