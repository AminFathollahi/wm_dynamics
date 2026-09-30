"""Re-evaluate affected behavior cells with serial-structure-preserving permutation sensitivities."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for subdirectory in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / subdirectory))

from corpus_sessions import data_root  # noqa: E402
from provenance import _json_safe, git_commit  # noqa: E402
from run_within_session_permutation_control import _load_multi_object_corpus, _qualifying_levels_from_arrays, within_session_permutation_test
from info_decoding import _blocks_for_levels, _circular_outcome_shift, _circular_residual_shift
from corpus_sessions import _reachable_sessions as _panichello_reachable_sessions
from corpus_sessions import MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION
from corpus_sessions import _session_arrays as rate_free_session_arrays

OUTPUT_PATH = ROOT / "results" / "empirical_trial_exchangeability_sensitivity.json"
N_PERMUTATIONS = 10000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lag1(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or np.std(values[:-1]) == 0.0 or np.std(values[1:]) == 0.0:
        return None
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def _series_diagnostics(session_blocks: list[list[dict]]) -> dict:
    predictor_lag1, outcome_lag1, outcome_residual_lag1 = [], [], []
    for blocks in session_blocks:
        for block in blocks:
            for target, values in ((predictor_lag1, block["predictor"]), (outcome_lag1, block["outcome"])):
                value = _lag1(values)
                if value is not None:
                    target.append(value)
            if block["controls"]:
                design = np.column_stack([np.ones(block["n"]), *block["controls"]])
                residual = block["outcome"] - design @ np.linalg.lstsq(design, block["outcome"], rcond=None)[0]
                value = _lag1(residual)
                if value is not None:
                    outcome_residual_lag1.append(value)
    return {
        "n_blocks": sum(len(blocks) for blocks in session_blocks),
        "predictor_lag1_median": float(np.median(predictor_lag1)) if predictor_lag1 else None,
        "outcome_lag1_median": float(np.median(outcome_lag1)) if outcome_lag1 else None,
        "outcome_residual_lag1_median": (
            float(np.median(outcome_residual_lag1)) if outcome_residual_lag1 else None
        ),
    }


def _evaluate(name: str, session_blocks: list[list[dict]], original: dict, n_perm: int) -> dict:
    uses_partial = any(block["controls"] for blocks in session_blocks for block in blocks)
    shuffle = _circular_residual_shift if uses_partial else _circular_outcome_shift
    result = within_session_permutation_test(
        session_blocks,
        f"empirical_trial_exchangeability_sensitivity|{name}",
        n_perm=n_perm,
        shuffle_block=shuffle,
    )
    original_statistic = original.get("observed_pooled_statistic")
    reproduced = (
        result.get("status") == "computed"
        and original_statistic is not None
        and abs(result["observed_pooled_statistic"] - original_statistic) < 1e-12
    )
    return {
        "status": "computed" if reproduced else "void_reproduction_gate_failed",
        "reproduces_original_observed_statistic": reproduced,
        "original_unrestricted_shuffle": original,
        "serial_structure_sensitivity": result,
        "sensitivity_method": "circular_residual_shift" if uses_partial else "circular_outcome_shift",
        "series_diagnostics": _series_diagnostics(session_blocks),
        "interpretation": (
            "survives_the_serial_structure_sensitivity"
            if reproduced and result.get("significant") else
            "does_not_survive_the_serial_structure_sensitivity"
            if reproduced else "not_interpretable"
        ),
    }


def build(n_perm: int) -> dict:
    source_path = ROOT / "results" / "within_session_permutation_control.json"
    calibration_path = ROOT / "results" / "trial_exchangeability_calibration.json"
    source = json.loads(source_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    rows_arrays, zero_drop = _load_multi_object_corpus()

    cells: dict[str, dict] = {}
    pooled_blocks = {"raw": [], "joint_partial_controlling_spike_count_and_trial_index": []}
    level2_blocks = []
    for arrays in rows_arrays.values():
        levels = _qualifying_levels_from_arrays(arrays)
        if levels:
            pooled_blocks["raw"].append(_blocks_for_levels(arrays, levels, "swap_primary", ()))
            pooled_blocks["joint_partial_controlling_spike_count_and_trial_index"].append(
                _blocks_for_levels(arrays, levels, "swap_primary", ("spike_count", "trial_index"))
            )
        mask = arrays["item_count"] == 2.0
        if int(mask.sum()) >= MIN_TRIALS_FOR_BEHAVIOURAL_CORRELATION:
            level2_blocks.append([{
                "n": int(mask.sum()), "predictor": arrays["deviation"][mask],
                "outcome": arrays["swap_primary"][mask], "controls": [],
            }])

    source_cells = source["cells"]
    cells["swap_pooled_within_item_count_level"] = {
        statistic: _evaluate(
            f"swap_pooled|{statistic}", blocks,
            source_cells["swap_pooled_within_item_count_level"]["within_session_permutation_control_valid"][statistic],
            n_perm,
        )
        for statistic, blocks in pooled_blocks.items()
    }
    cells["swap_level2_only"] = {
        "raw": _evaluate(
            "swap_level2|raw", level2_blocks,
            source_cells["swap_level2_only"]["within_session_permutation_control_valid"]["raw"], n_perm,
        )
    }

    rate_blocks = {"raw": [], "joint_partial_controlling_spike_count_and_trial_index": []}
    for path in _panichello_reachable_sessions(data_root()):
        arrays = rate_free_session_arrays(path)
        if arrays is None:
            continue
        n = len(arrays["is_corr"])
        rate_blocks["raw"].append([{
            "n": n, "predictor": arrays["deviation"], "outcome": arrays["is_corr"], "controls": [],
        }])
        rate_blocks["joint_partial_controlling_spike_count_and_trial_index"].append([{
            "n": n, "predictor": arrays["deviation"], "outcome": arrays["is_corr"],
            "controls": [arrays["spike_count"], arrays["trial_index"]],
        }])
    cells["rate_free_headline"] = {
        statistic: _evaluate(
            f"rate_free|{statistic}", blocks,
            source_cells["rate_free_headline"]["within_session_permutation_control_valid"][statistic], n_perm,
        )
        for statistic, blocks in rate_blocks.items()
    }

    return {
        "analysis_id": "empirical_trial_exchangeability_sensitivity",
        "schema_version": "1.0.0",
        "status": "complete",
        "source_artifact": {"path": "results/within_session_permutation_control.json", "sha256": _sha256(source_path)},
        "calibration_artifact": {"path": "results/trial_exchangeability_calibration.json", "sha256": _sha256(calibration_path)},
        "producer": {"path": "scripts/run_empirical_trial_exchangeability_sensitivity.py", "sha256": _sha256(Path(__file__)), "git_commit": git_commit(ROOT)},
        "n_permutations": n_perm,
        "method_scope": (
            "Circular shifts preserve each tested outcome or reduced-model residual series but assume circular "
            "stationarity of the finite within-block sequence. Item-count filtering preserves trial order but not "
            "the original spacing between retained trials. These are calibrated sensitivities under the simulated "
            "regimes, not universal exact tests."
        ),
        "pre_cue_cell_scope": (
            "The pre-cue cell uses independent reported-versus-other label exchange within each trial and does not "
            "permute outcomes across trials; the generic serial-order calibration does not apply to that null."
        ),
        "zero_drop_accounting_multi_object": zero_drop,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = build(args.permutations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(output), indent=2, allow_nan=False, default=float) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
