#!/usr/bin/env python3
"""compute_admission_matrix_margins.py -- convert
results/observability_and_power_census.json's admission matrix from a
binary admitted/excluded verdict to a verdict that also states the margin
by which each cell clears (or fails to clear) its floor.

Every number this script writes is already on disk: each admitted or
excluded cell-observable entry already carries measured_value and
required_value from the census's own build_admission_matrix. This script
reads the existing artifact, computes a ratio and a named band from those
two numbers, adds the small-worldness margin test case and the margin-aware
single-unit-versus-LFP modality comparison, and writes the result back to
the same path with the schema version bumped. No admission-matrix row is
refit and no corpus is touched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_observability_and_power_census as census  # noqa: E402
from provenance import canonical_json  # noqa: E402

INPUT_OUTPUT_PATH = census.OUTPUT_PATH
SCHEMA_VERSION_BEFORE_MARGINS = "2.0.0"
SCHEMA_VERSION_WITH_MARGINS = "2.1.0"


def add_margins_to_existing_census(path: Path = INPUT_OUTPUT_PATH) -> dict:
    payload = json.loads(path.read_text())
    if payload["schema_version"] != SCHEMA_VERSION_BEFORE_MARGINS:
        raise ValueError(
            f"expected schema_version {SCHEMA_VERSION_BEFORE_MARGINS!r} on {path}, found "
            f"{payload['schema_version']!r} -- refusing to guess whether margins were already added"
        )

    matrix = payload["admission_matrix"]
    band_definitions = census.predeclare_admission_margin_bands()
    n_margins_added = census.add_admission_margins(matrix)
    recount = census.summarize_admission_margin_recount(matrix)
    small_worldness_case = census.small_worldness_margin_test_case(matrix)
    modality_rule = census.predeclare_margin_aware_modality_rule()
    modality_result = census.evaluate_margin_aware_modality_rule(matrix, modality_rule)

    payload["schema_version"] = SCHEMA_VERSION_WITH_MARGINS
    payload["admission_matrix"] = matrix
    payload["admission_margin_bands"] = band_definitions
    payload["admission_margin_recount"] = recount
    payload["small_worldness_margin_test_case"] = small_worldness_case
    payload["margin_aware_modality_predeclared_rule"] = modality_rule
    payload["margin_aware_modality_result"] = modality_result
    payload["admission_margin_addition"] = {
        "n_cell_observable_entries_receiving_a_margin": n_margins_added,
        "schema_version_before_margins": SCHEMA_VERSION_BEFORE_MARGINS,
        "note": (
            "every admitted or excluded cell-observable entry's status, measured_value and required_value "
            "field is byte-identical to the pre-margin artifact; the margin field is additive"
        ),
    }

    path.write_text(canonical_json(payload))
    return payload


def main() -> None:
    payload = add_margins_to_existing_census()
    recount = payload["admission_margin_recount"]
    sw = payload["small_worldness_margin_test_case"]
    print(json.dumps({
        "admission_margin_recount": recount,
        "small_worldness_margin_test_case": {k: v for k, v in sw.items() if k != "at_floor_cells"},
        "margin_aware_modality_result": {
            width: {"branch": r["branch"], "summed_rank_difference_lfp_minus_unit": r["summed_rank_difference_lfp_minus_unit"]}
            for width, r in payload["margin_aware_modality_result"].items()
        },
    }, indent=2))
    print(f"wrote {INPUT_OUTPUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
