#!/usr/bin/env python3
"""One-off repair: every `lambda_regional_ordering`
block in results/ was built by checking only ``state_space...median is not
None`` (true even at n_patients=1, since a bootstrap CI of one repeated
value is still numerically well-defined), not ``n_patients >= 2``. That let
single-patient point estimates rank first in region_stratified_drift_000469
.json (amygdala, vmpfc), region_stratified_drift_000574.json (hippocampus),
and region_stratified_drift_001187_000673.json's content_axis_battery
(pre_sma), the worked example that surfaced the defect.

The fitting scripts (run_human_drift_spine_000469.py/000574.py/
001187_000673.py) already had their group-statistic builders fixed to gate
at n_patients >= 2 going forward.
This script repairs the three ALREADY-WRITTEN artifacts in place, from data
already present in each file (no refit): it restricts each ordering block's
`region_lambda_state_space_median` / `region_order_fastest_to_slowest` to
structures whose `state_space_lambda_identified_mean.n_patients >= 2`, and
records the excluded structures explicitly rather than dropping them
silently.

Run:
    conda run -n wm_dynamics python scripts/fix_lambda_ordering_underpowered_leakage.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from provenance import canonical_json  # noqa: E402

RESULTS = ROOT / "results"
MIN_PATIENTS_FOR_ORDERING = 2


def _rebuild_ordering(regions: dict, region_names: list[str]) -> dict:
    included, excluded = {}, {}
    for region in region_names:
        if region == "pooled":
            continue  # the superseded chimeric baseline, not a peer anatomical structure
        block = regions.get(region)
        if block is None:
            continue
        group = block.get("group")
        if not isinstance(group, dict):
            continue
        state_space = group.get("state_space_lambda_identified_mean")
        if not isinstance(state_space, dict) or state_space.get("median") is None:
            continue
        n_patients = state_space.get("n_patients", 0)
        if n_patients >= MIN_PATIENTS_FOR_ORDERING:
            included[region] = state_space["median"]
        else:
            excluded[region] = {"median": state_space["median"], "n_patients": n_patients}
    if len(included) >= 2:
        ordered = sorted(included, key=included.get, reverse=True)
        return {
            "status": "estimable",
            "region_lambda_state_space_median": included,
            "region_order_fastest_to_slowest": ordered,
            "excluded_underpowered": excluded,
            "note": (
                "Rank-based ordering of per-region state-space lambda medians, restricted to "
                f"structures with >= {MIN_PATIENTS_FOR_ORDERING} identified patients for this "
                "specific metric -- previously any structure with a computed "
                "median was included regardless of n, letting single-patient point estimates rank "
                "first). excluded_underpowered lists structures with a median but fewer than "
                f"{MIN_PATIENTS_FOR_ORDERING} patients, reported not silently dropped. Not a fitted "
                "hierarchy model."
            ),
        }
    return {
        "status": "non_identified",
        "reason": f"fewer than 2 regions had an identifiable state-space lambda median at n >= {MIN_PATIENTS_FOR_ORDERING} patients",
        "excluded_underpowered": excluded,
    }


def fix_000469() -> bool:
    path = RESULTS / "region_stratified_drift_000469.json"
    d = json.loads(path.read_text())
    region_names = list(d["regions"].keys())
    new_ordering = _rebuild_ordering(d["regions"], region_names)
    if d.get("lambda_regional_ordering") == new_ordering:
        return False
    new_ordering["mtl_faster_than_medial_frontal"] = None
    if new_ordering["status"] == "estimable":
        mtl = [r for r in new_ordering["region_order_fastest_to_slowest"] if r == "hippocampus"]
        frontal = [r for r in new_ordering["region_order_fastest_to_slowest"] if r in ("pre_sma", "dacc", "vmpfc")]
        if mtl and frontal:
            medians = new_ordering["region_lambda_state_space_median"]
            new_ordering["mtl_faster_than_medial_frontal"] = bool(medians[mtl[0]] > max(medians[r] for r in frontal))
    d["lambda_regional_ordering"] = new_ordering
    path.write_text(canonical_json(d))
    return True


def fix_000574() -> bool:
    path = RESULTS / "region_stratified_drift_000574.json"
    d = json.loads(path.read_text())
    region_names = list(d["regions"].keys())
    new_ordering = _rebuild_ordering(d["regions"], region_names)
    if d.get("lambda_regional_ordering") == new_ordering:
        return False
    d["lambda_regional_ordering"] = new_ordering
    path.write_text(canonical_json(d))
    return True


def fix_001187_000673() -> bool:
    path = RESULTS / "region_stratified_drift_001187_000673.json"
    d = json.loads(path.read_text())
    battery = d.get("content_axis_battery")
    if battery is None:
        return False
    region_names = list(battery["regions"].keys())
    new_ordering = _rebuild_ordering(battery["regions"], region_names)
    if battery.get("lambda_regional_ordering") == new_ordering:
        return False
    battery["lambda_regional_ordering"] = new_ordering
    path.write_text(canonical_json(d))
    return True


def main() -> None:
    changed = {
        "region_stratified_drift_000469.json": fix_000469(),
        "region_stratified_drift_000574.json": fix_000574(),
        "region_stratified_drift_001187_000673.json (content_axis_battery)": fix_001187_000673(),
    }
    print(json.dumps({"changed": changed}, indent=2))


if __name__ == "__main__":
    main()
