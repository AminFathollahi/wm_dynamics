#!/usr/bin/env python3
"""Build the model-based control payload or record why it is not identified."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import canonical_json, git_commit, sha256_file  # noqa: E402

SPINE = ROOT / "results" / "human_drift_spine_000469.json"
BEHAVIOR = ROOT / "results" / "human_drift_behavior_000469.json"
OUTPUT = ROOT / "results" / "drift_control_payload_000469.json"


def main() -> None:
    spine = json.loads(SPINE.read_text())
    behavior = json.loads(BEHAVIOR.read_text())
    tolerance_identified = behavior.get("tolerance_status") == "identified"
    lambda_identifiability = spine["group"]["identifiability"]
    if not tolerance_identified:
        output = {
            "schema_version": "1.0.0", "analysis_id": "drift_control_payload_dandi000469",
            "status": "nonidentified",
            "reason": "probe displacement does not predict memory error with a positive bounded slope, so no empirical tolerance threshold exists",
            "code_commit": git_commit(ROOT),
            "inputs": {
                "spine": {"path": str(SPINE.relative_to(ROOT)), "hash": sha256_file(SPINE)},
                "behavior": {"path": str(BEHAVIOR.relative_to(ROOT)), "hash": sha256_file(BEHAVIOR)},
            },
            "tolerance_ellipsoid": {
                "status": "nonidentified", "criterion_error_probability": 0.25,
                "reason": "hierarchical drift slope interval includes zero",
            },
            "passive_prediction_falsification": {
                "status": "nonidentified",
                "reason": "requires an empirical tolerance and identified session-level D/lambda",
            },
            "control_cost": {
                "status": "nonidentified",
                "reason": "required gain g cannot be computed without a tolerance; most fold-level lambda estimates also fail the sampling-window identifiability rule",
            },
            "lambda_identifiability": lambda_identifiability,
            "interpretation": "This dataset supports held-out trial-specific temporal dependence, not a behaviorally calibrated control requirement.",
            "prohibited_interpretation": "Do not call this achieved control, controllability, steering, or a tolerance bound.",
        }
    else:
        raise RuntimeError("identified-tolerance computation is not implemented; refusing a silent partial payload")
    OUTPUT.write_text(canonical_json(output))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

