"""Regression check for null aggregate fields beside computed status without reasons:
when an aggregate has status:computed, every conclusion-bearing numeric field must either
have a value or have an adjacent reason field explaining why it could not be computed.

No frameworks, no fixtures -- plain assert-based checks, runnable directly:
    python tests/test_robustness_aggregate_completeness.py
or picked up by pytest as ordinary top-level test_* functions.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = ROOT / "results" / "state_space_estimation_robustness.json"


def _walk_aggregates(obj, path=""):
    """Recursively yield all aggregate objects found in the artifact."""
    if isinstance(obj, dict):
        if "aggregate" in obj and isinstance(obj["aggregate"], dict):
            yield (path + ".aggregate", obj["aggregate"])
        for key, val in obj.items():
            yield from _walk_aggregates(val, path + "." + key if path else key)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            yield from _walk_aggregates(item, path + f"[{i}]")


def test_no_null_conclusion_fields_without_reasons():
    """Every aggregate with status:computed must not have bare nulls for numeric conclusions."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")

    data = json.loads(ARTIFACT_PATH.read_text())

    conclusion_fields = {
        "fraction_significant_p_below_0p05",
        "mean_effect_size",
        "mean_rho",
        "alignment_excess_over_random",
    }

    found_issues = []

    for agg_path, agg in _walk_aggregates(data):
        if agg.get("status") != "computed":
            continue

        for field in conclusion_fields:
            if field not in agg:
                continue

            val = agg[field]
            if val is None:
                reason_field = field.replace("_p_below_0p05", "_not_computable_reason").replace("_", "_") if "p_below" in field else f"{field}_not_computable_reason"

                # Try common reason field name patterns
                reason_found = (
                    reason_field in agg or
                    f"{field}_reason" in agg or
                    f"{field}_reason_not_computable" in agg
                )

                if not reason_found:
                    found_issues.append(
                        f"{agg_path}: {field} is null with status:computed but no reason field"
                    )

    if found_issues:
        raise AssertionError(
            "Found bare null conclusion fields without reason explanations:\n  " +
            "\n  ".join(found_issues)
        )

    print("PASS: all null conclusion fields beside computed status have reason fields.")


def test_memorandum_coding_subspace_vs_deviation_fraction_computed():
    """Specific regression check: the memorandum_coding_subspace_vs_deviation claim's
    fraction_significant_p_below_0p05 field must be computed from per-cell p-values."""
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(f"Artifact not found: {ARTIFACT_PATH}")

    data = json.loads(ARTIFACT_PATH.read_text())

    # Check rung_three entry
    rung_three = data.get("rung_three", {})
    entry = rung_three.get("memorandum_coding_subspace_vs_deviation", {})
    agg = entry.get("aggregate", {})

    if agg.get("status") == "computed":
        frac = agg.get("fraction_significant_p_below_0p05")
        if frac is None:
            raise AssertionError(
                "rung_three.memorandum_coding_subspace_vs_deviation.aggregate: "
                "fraction_significant_p_below_0p05 is null with status:computed"
            )

        # Verify it's computed correctly from cells
        cells = entry.get("cells", [])
        name = "memorandum_coding_subspace_vs_deviation"
        pvals = [c.get(name, {}).get("p_value") for c in cells
                 if c.get("status") == "computed"
                 and c.get(name, {}).get("p_value") is not None]

        if len(pvals) > 0:
            expected_frac = sum(1 for p in pvals if p < 0.05) / len(pvals)
            if abs(frac - expected_frac) > 1e-10:
                raise AssertionError(
                    f"rung_three.memorandum_coding_subspace_vs_deviation.aggregate: "
                    f"fraction_significant_p_below_0p05 is {frac} but expected {expected_frac} "
                    f"({sum(1 for p in pvals if p < 0.05)} of {len(pvals)} cells)"
                )

    print("PASS: memorandum_coding_subspace_vs_deviation fraction_significant is correctly computed.")


if __name__ == "__main__":
    test_no_null_conclusion_fields_without_reasons()
    test_memorandum_coding_subspace_vs_deviation_fraction_computed()
    print("ALL CHECKS PASSED")
