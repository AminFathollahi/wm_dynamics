from __future__ import annotations

import copy
import json
import os
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
from run_component_identity_subspace_atlas import _panichello_session_inputs  # noqa: E402
from statistics import stable_seed  # noqa: E402
import run_within_animal_component_identity as identity_analysis  # noqa: E402

OUTPUT_PATH = ROOT / "results" / "pfc4_content_cell_checks.json"
DELIVERED_PATH = ROOT / "results" / "within_animal_component_identity.json"
SCHEMA_VERSION = "1.0"
VERSION = "2026-09-18"
SEED_NAMESPACE = f"pfc4_content_cell_checks|{VERSION}"
PFC4_CORPUS = identity_analysis.PFC4_CORPUS
UNIT_FLOOR = 5
N_ASSOCIATION_PERM = 200
N_BOOT = 2000
DOCUMENTED_F1_VALUES_HZ = (10.0, 14.0, 18.0, 24.0, 30.0, 34.0)

SCOPE = (
    "two pre-declared checks on the pfc4_macaque_somatosensory memorandum_content cell for rr014, the "
    "only within-animal content cell that has cleared in any macaque animal in this project. Neither "
    "check reclassifies or rewrites the delivered within-animal component identity artifact; its "
    "verdicts stand as fired. Check one asks whether the rr014 clear survives restricting pfc4 sessions "
    "to a higher simultaneous-unit floor than this corpus's own loader admits. Check two measures, not "
    "asserts, whether the pfc4 memorandum (a vibrotactile frequency magnitude) is separable from total "
    "spike count in the way the other two macaque corpora's spatial or object memoranda are."
)

DECISION_RULE_CHECK_1 = (
    "Pre-declared before either fit was run. The pfc4_macaque_somatosensory memorandum_content cell for "
    "each animal is re-pooled twice: once at the delivered loader's own floor (at least 2 simultaneously "
    "recorded units per session) and once restricted to sessions carrying at least "
    f"{UNIT_FLOOR} simultaneously recorded units, using the identical per-session fits, cross-fitting and "
    "permutation draws already computed for the delivered artifact and the identical finalisation rule "
    "(false-discovery-rate correction across every computed primary cell in that same artifact family, "
    "AND a whole-session cluster bootstrap interval excluding zero) applied separately to each of the two "
    "fits. If the resulting verdict differs between the two fits, the cell is reported UNSTABLE, not as a "
    "result. If the verdict holds at both, it holds. A cell with fewer than "
    f"{identity_analysis.MIN_SESSIONS_PER_ANIMAL} surviving sessions after the unit restriction is "
    "reported not_computable with its own session count, never as a null or as stability."
)

DECISION_RULE_CHECK_2 = (
    "Measures the association between each corpus's own memorandum variable and that trial's total spike "
    "count (the same trial-level quantity the identity estimand's own gain_total_spike_count candidate "
    "uses), as the fraction of total-spike-count variance a linear fit of the memorandum construction "
    "explains, corrected for the fit's own mechanical inflation by subtracting the mean of the identical "
    "statistic computed under label permutation -- because the three corpora's memorandum constructions "
    "have different column counts (one column for pfc4's frequency, two for the Watters angle, one column "
    "per discrete location for Panichello's class-mean construction), an uncorrected in-sample fraction of "
    "variance explained is not comparable across them on its own. Reported per session, as a per-animal "
    "distribution, and as a whole-session cluster bootstrap summary per animal. No clear/bound verdict is "
    "assigned; this is a measurement, not a hypothesis test."
)


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


# ── Check 1: unit-count sensitivity ──────────────────────────────────────────

def _unit_counts(root: Path) -> tuple[dict[str, int], dict[str, dict]]:
    per_session: dict[str, int] = {}
    per_animal: dict[str, list[int]] = {}
    for entry in iter_pfc4(root):
        per_session[entry["session"]] = entry["n_units"]
        per_animal.setdefault(entry["patient"], []).append(entry["n_units"])
    histogram = {
        animal: {
            "n_sessions": len(values),
            "unit_count_histogram": {str(k): int(sum(1 for v in values if v == k)) for k in range(2, 8)},
        }
        for animal, values in per_animal.items()
    }
    return per_session, histogram


def _grouped_by_animal(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(record["independent_unit"], []).append(record)
    return grouped


def _pool_filtered(records: list[dict], unit_counts: dict[str, int], min_units: int, candidate: str, seed: str) -> dict:
    cells = []
    for record in records:
        if record.get("status") != "computed":
            continue
        n_units = unit_counts.get(record["session"])
        if n_units is None or n_units < min_units:
            continue
        cells.append(record["candidates"][candidate])
    pooled = identity_analysis._pool_within_animal(cells, seed)
    pooled["n_sessions_at_this_floor"] = len(cells)
    return pooled


def _finalize_with_pfc4(delivered_per_corpus: dict, gain_by_animal: dict, content_by_animal: dict) -> dict:
    per_corpus = copy.deepcopy(delivered_per_corpus)
    per_corpus[PFC4_CORPUS]["gain_total_spike_count"] = gain_by_animal
    per_corpus[PFC4_CORPUS]["memorandum_content"] = {
        animal: {**cell, "content_construction": identity_analysis.CONTENT_CONSTRUCTION_NOTE[PFC4_CORPUS]}
        for animal, cell in content_by_animal.items()
    }
    return identity_analysis._finalize(per_corpus)[PFC4_CORPUS]


def _status_or_verdict(cell: dict) -> str:
    if cell.get("status") != "computed":
        return cell.get("status", "not_computable")
    return cell.get("verdict", "computed_no_verdict")


def unit_count_sensitivity(root: Path, delivered: dict) -> dict:
    identity = delivered["identity"]
    n_perm = identity["n_perm"]
    records = identity_analysis._pfc4_sessions(root, n_perm, identity)
    unit_counts, histogram = _unit_counts(root)
    grouped = _grouped_by_animal(records)

    full_gain, full_content, restricted_gain, restricted_content = {}, {}, {}, {}
    for animal, animal_records in grouped.items():
        full_gain[animal] = _pool_filtered(
            animal_records, unit_counts, 2, "gain_total_spike_count",
            f"{PFC4_CORPUS}|{animal}|gain_total_spike_count|bootstrap",
        )
        full_content[animal] = _pool_filtered(
            animal_records, unit_counts, 2, "memorandum_content",
            f"{PFC4_CORPUS}|{animal}|memorandum_content|bootstrap",
        )
        restricted_gain[animal] = _pool_filtered(
            animal_records, unit_counts, UNIT_FLOOR, "gain_total_spike_count",
            f"{SEED_NAMESPACE}|{animal}|gain_total_spike_count|min_units_{UNIT_FLOOR}",
        )
        restricted_content[animal] = _pool_filtered(
            animal_records, unit_counts, UNIT_FLOOR, "memorandum_content",
            f"{SEED_NAMESPACE}|{animal}|memorandum_content|min_units_{UNIT_FLOOR}",
        )

    full_result = _finalize_with_pfc4(delivered["summary"]["per_corpus"], full_gain, full_content)
    restricted_result = _finalize_with_pfc4(delivered["summary"]["per_corpus"], restricted_gain, restricted_content)

    reproduction_differences = []
    for animal, delivered_cell in delivered["summary"]["per_corpus"][PFC4_CORPUS]["memorandum_content"].items():
        recomputed_cell = full_result["memorandum_content"][animal]
        for key in ("mean_alignment_above_null", "p_value", "minimum_detectable_difference_80pct_power"):
            if abs(float(delivered_cell[key]) - float(recomputed_cell[key])) > 1e-9:
                reproduction_differences.append(f"{animal}.{key}: {delivered_cell[key]} vs {recomputed_cell[key]}")

    content_cells = {}
    for animal in sorted(restricted_result["memorandum_content"]):
        full_cell = full_result["memorandum_content"][animal]
        restricted_cell = restricted_result["memorandum_content"][animal]
        full_status = _status_or_verdict(full_cell)
        restricted_status = _status_or_verdict(restricted_cell)
        if full_cell.get("status") != "computed" or restricted_cell.get("status") != "computed":
            stability, reported = "not_computable", "not_computable"
        elif full_status == restricted_status:
            stability, reported = "stable", full_status
        else:
            stability, reported = "unstable", "unstable"
        content_cells[animal] = {
            "full_floor_min_units": 2,
            "restricted_floor_min_units": UNIT_FLOOR,
            "full_floor_fit": full_cell,
            "restricted_floor_fit": restricted_cell,
            "verdict_stability": stability,
            "reported_verdict": reported,
        }

    gain_cells = {
        animal: {
            "full_floor_fit": full_result["gain_total_spike_count"][animal],
            "restricted_floor_fit": restricted_result["gain_total_spike_count"][animal],
        }
        for animal in sorted(restricted_result["gain_total_spike_count"])
    }

    return {
        "decision_rule": DECISION_RULE_CHECK_1,
        "unit_count_histogram_per_animal": histogram,
        "unit_floor_restricted_to": UNIT_FLOOR,
        "memorandum_content": content_cells,
        "gain_total_spike_count_internal_reference": gain_cells,
        "full_floor_reproduction_check_against_delivered_artifact": {
            "status": "reproduced_exactly" if not reproduction_differences else "reproduction_mismatch",
            "differences": reproduction_differences,
        },
    }


# ── Check 2: parametric magnitude confound ───────────────────────────────────

def _r_squared_above_null(design_columns, y, n_perm: int, rng: np.random.Generator) -> dict | None:
    X = np.asarray(design_columns, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[valid], y[valid]
    if len(y) < 6:
        return None

    def _fit(target: np.ndarray) -> float | None:
        design = np.column_stack([X, np.ones(len(target))])
        coef, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coef
        ss_tot = float(np.sum((target - target.mean()) ** 2))
        if ss_tot <= 0:
            return None
        return 1.0 - float(np.sum(residual ** 2)) / ss_tot

    observed = _fit(y)
    if observed is None:
        return None
    null = np.array([_fit(rng.permutation(y)) for _ in range(n_perm)], dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return None
    return {
        "n_trials": int(len(y)),
        "r_squared": float(observed),
        "r_squared_null_mean": float(null.mean()),
        "r_squared_above_null": float(observed - null.mean()),
    }


def _pfc4_association(root: Path) -> list[dict]:
    rows = []
    for entry in iter_pfc4(root):
        activity = np.asarray(entry["counts"]).sum(axis=2)
        total = activity.sum(axis=1)
        rng = np.random.default_rng(_seed(f"pfc4_association|{entry['session']}"))
        result = _r_squared_above_null(entry["f1_hz"], total, N_ASSOCIATION_PERM, rng)
        if result is not None:
            rows.append({"animal": entry["patient"], "session": entry["session"], **result})
    return rows


def _panichello_association(root: Path) -> list[dict]:
    rows = []
    for session_id, reason, core, categorical, continuous, counts_all in _panichello_session_inputs(root):
        if core is None:
            continue
        cue = np.asarray(categorical["memorandum_content"])
        total = core["spike_count"]
        onehot = (cue[:, None] == np.unique(cue)[None, :]).astype(float)
        rng = np.random.default_rng(_seed(f"panichello_association|{session_id}"))
        result = _r_squared_above_null(onehot, total, N_ASSOCIATION_PERM, rng)
        if result is not None:
            animal = independent_unit("panichello_2024_macaque_lPFC", session_id)
            rows.append({"animal": animal, "session": session_id, **result})
    return rows


def _watters_association(root: Path) -> list[dict]:
    behaviour = watters_behaviour(root)
    rows = []
    for animal, session_date, variant in watters_session_dates(root):
        session = load_watters_session(root, animal, session_date, behaviour)
        if session.get("status") != "loaded":
            continue
        total = session["counts"].sum(axis=(1, 2))
        theta = np.asarray(session["cued_theta"], dtype=float)
        design = np.column_stack([np.cos(theta), np.sin(theta)])
        session_id = f"{animal}_{session_date}_{variant}"
        rng = np.random.default_rng(_seed(f"watters_association|{session_id}"))
        result = _r_squared_above_null(design, total, N_ASSOCIATION_PERM, rng)
        if result is not None:
            rows.append({"animal": animal, "session": session_id, **result})
    return rows


def _cluster_summary(rows: list[dict], seed_prefix: str) -> dict:
    by_animal: dict[str, list[float]] = {}
    for row in rows:
        by_animal.setdefault(row["animal"], []).append(row["r_squared_above_null"])
    summary = {}
    for animal, values in by_animal.items():
        array = np.asarray(values, dtype=float)
        rng = np.random.default_rng(_seed(f"{seed_prefix}|{animal}|cluster_bootstrap"))
        boot = np.array([
            array[rng.integers(0, len(array), len(array))].mean() for _ in range(N_BOOT)
        ])
        summary[animal] = {
            "n_sessions": int(len(array)),
            "mean_r_squared_above_null": float(array.mean()),
            "median_r_squared_above_null": float(np.median(array)),
            "min_r_squared_above_null": float(array.min()),
            "max_r_squared_above_null": float(array.max()),
            "session_cluster_bootstrap_interval_95pct": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        }
    return summary


def _f1_value_discrepancy(root: Path) -> dict:
    documented = set(DOCUMENTED_F1_VALUES_HZ)
    per_session = []
    union_values: set[float] = set()
    for entry in iter_pfc4(root):
        values = set(np.round(entry["f1_hz"], 6).tolist())
        union_values |= values
        per_session.append({
            "animal": entry["patient"], "session": entry["session"],
            "n_distinct_f1_values": len(values),
            "non_documented_values_hz": sorted(values - documented),
        })
    by_animal: dict[str, dict] = {}
    for row in per_session:
        stats = by_animal.setdefault(row["animal"], {"n_sessions": 0, "n_sessions_with_non_documented_values": 0})
        stats["n_sessions"] += 1
        if row["non_documented_values_hz"]:
            stats["n_sessions_with_non_documented_values"] += 1
    return {
        "documented_values_hz": sorted(documented),
        "union_distinct_values_hz": sorted(union_values),
        "n_union_distinct_values": len(union_values),
        "value_range_hz": [min(union_values), max(union_values)],
        "by_animal": by_animal,
        "n_sessions_total": len(per_session),
        "n_sessions_with_non_documented_values_total": sum(
            1 for row in per_session if row["non_documented_values_hz"]
        ),
    }


def parametric_magnitude_confound(root: Path) -> dict:
    pfc4_rows = _pfc4_association(root)
    panichello_rows = _panichello_association(root)
    watters_rows = _watters_association(root)
    return {
        "decision_rule": DECISION_RULE_CHECK_2,
        "per_session": {
            PFC4_CORPUS: pfc4_rows,
            "panichello_2024_macaque_lPFC": panichello_rows,
            "watters_2026_macaque_multi_object": watters_rows,
        },
        "per_animal_cluster_summary": {
            PFC4_CORPUS: _cluster_summary(pfc4_rows, "pfc4_association"),
            "panichello_2024_macaque_lPFC": _cluster_summary(panichello_rows, "panichello_association"),
            "watters_2026_macaque_multi_object": _cluster_summary(watters_rows, "watters_association"),
        },
        "f1_value_discrepancy": _f1_value_discrepancy(root),
    }


def main() -> None:
    started = time.time()
    root = data_root()
    delivered = json.loads(DELIVERED_PATH.read_text())

    output = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "scope": SCOPE,
        "data_root": str(root),
        "delivered_artifact_reference": str(DELIVERED_PATH.relative_to(ROOT)),
    }
    _write(OUTPUT_PATH, output)

    output["unit_count_sensitivity"] = unit_count_sensitivity(root, delivered)
    _write(OUTPUT_PATH, output)

    output["parametric_magnitude_confound"] = parametric_magnitude_confound(root)
    output["status"] = "complete"
    output["git_commit"] = git_commit(ROOT)
    output["wall_clock_s"] = time.time() - started
    _write(OUTPUT_PATH, output)


if __name__ == "__main__":
    main()
