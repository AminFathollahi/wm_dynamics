"""Structural test: a d_perm slope is never reported without its two
component slopes (r_obs and the permutation-null r_null) present as
siblings in the same result block.

This is a standing rule in this project's review process (any claim about a
d_perm level, slope or sign must carry the observed correlation and the
permutation null it is measured against, because d_perm = r_obs - r_null is
an arithmetic contrast whose arithmetic alone explains nothing) -- easy to
satisfy by construction when a field is written, easy to silently break
later when a result block is refactored or a new one is added and someone
forgets the sibling fields. :func:`find_d_perm_slope_violations` catches
that mechanically instead of relying on a reviewer noticing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def find_d_perm_slope_violations(obj, path: str = "") -> list[str]:
    """Recursively walks a JSON-like nested dict/list structure and returns
    the path of every dict that reports a d_perm slope-test result (a key
    literally named "d_perm" whose value is itself a dict carrying a "test"
    sub-dict with a "mean_value") without both of its two component
    correlations present as sibling keys in the SAME dict: a key
    recognisably the observed correlation ("r_obs") and a key recognisably
    the null it is measured against ("r_null" or "r_null_permutation").

    Only matches the segmented-slope-test result shape (a "d_perm" dict
    with a nested "test"/"mean_value") -- the shape
    scripts/run_state_shape_common_range.py's slope decomposition produces.
    A d_perm PER-LAG record that already carries plain "r_obs"/"r_null"
    float siblings (e.g. scripts/run_state_behavior_link.json's per_lag
    tables) is a different, already-compliant shape and is not what this
    checker is looking for; it targets the one place a slope summary could
    be written without its components, which is exactly where the defect
    this rule guards against actually occurred.
    """
    violations: list[str] = []
    if isinstance(obj, dict):
        d_perm_value = obj.get("d_perm")
        if (isinstance(d_perm_value, dict) and isinstance(d_perm_value.get("test"), dict)
                and "mean_value" in d_perm_value["test"]):
            has_r_obs = isinstance(obj.get("r_obs"), dict)
            has_r_null = isinstance(obj.get("r_null"), dict) or isinstance(obj.get("r_null_permutation"), dict)
            if not (has_r_obs and has_r_null):
                violations.append(path or "<root>")
        for key, value in obj.items():
            violations.extend(find_d_perm_slope_violations(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            violations.extend(find_d_perm_slope_violations(item, f"{path}[{i}]"))
    return violations


def test_compliant_block_has_no_violations():
    block = {
        "d_perm": {"test": {"mean_value": -0.1022}},
        "r_obs": {"test": {"mean_value": -0.1885}},
        "r_null_permutation": {"test": {"mean_value": -0.0863}},
    }
    assert find_d_perm_slope_violations(block) == []


def test_missing_r_null_sibling_is_flagged():
    block = {"d_perm": {"test": {"mean_value": -0.1022}}, "r_obs": {"test": {"mean_value": -0.1885}}}
    assert find_d_perm_slope_violations(block) == ["<root>"]


def test_missing_r_obs_sibling_is_flagged():
    block = {"d_perm": {"test": {"mean_value": -0.1022}}, "r_null": {"test": {"mean_value": -0.0863}}}
    assert find_d_perm_slope_violations(block) == ["<root>"]


def test_nested_violation_reports_its_path():
    obj = {"arm": {"range": {"d_perm": {"test": {"mean_value": 0.0998}}}}}
    assert find_d_perm_slope_violations(obj) == ["arm.range"]


def test_d_perm_without_a_test_summary_is_not_a_slope_and_is_not_flagged():
    """A plain per-lag record (d_perm as a bare float, or without a nested
    "test"/"mean_value") is not the slope-summary shape this rule targets
    and must not be flagged -- e.g. scripts/run_state_behavior_link.json's
    per_lag entries, which already carry r_obs/r_null as plain float
    siblings in a different, already-compliant shape."""
    obj = {"per_lag": {"3": {"r_obs": 0.5, "r_null": 0.3, "d_perm": 0.2}}}
    assert find_d_perm_slope_violations(obj) == []


def test_delivered_common_range_artifact_has_no_d_perm_slope_violations():
    """The real deliverable this rule protects: every d_perm slope block in
    results/state_shape_common_range.json (produced by
    scripts/run_state_shape_common_range.py) carries its r_obs and
    permutation-null component slopes as siblings, mechanically checked
    against the artifact actually on disk rather than only against a
    synthetic fixture."""
    artifact_path = Path(__file__).resolve().parents[1] / "results" / "state_shape_common_range.json"
    if not artifact_path.exists():
        return
    artifact = json.loads(artifact_path.read_text())
    violations = find_d_perm_slope_violations(artifact)
    assert violations == [], f"d_perm slope(s) reported without both component slopes at: {violations}"
