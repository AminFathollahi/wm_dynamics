"""Regression check for the silent-default registry-key defect: a missing or
mistyped dataset-registry key must raise, never silently be treated as "0
sessions" (a completed-looking empty analysis) or a fabricated null sitting
next to a status field that doesn't actually say why.

No frameworks, no fixtures -- plain assert-based checks, runnable directly:
    python tests/test_no_silent_registry_defaults.py
or picked up by pytest as ordinary top-level test_* functions.
"""
import importlib.util
import json
import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LOADER_MODULES = [
    "run_state_behavior_link",
    "run_state_latent_identity",
    "run_state_persistence",
    "run_state_content_link",
]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_helpers_raise_on_missing_registry_key() -> None:
    real_config = json.loads((ROOT / "config" / "datasets.json").read_text())
    assert "panichello_2024" in real_config["datasets"], (
        "fixture assumption broken: 'panichello_2024' is no longer in the real registry"
    )

    broken_config = json.loads(json.dumps(real_config))
    del broken_config["datasets"]["panichello_2024"]
    broken_text = json.dumps(broken_config)

    for name in LOADER_MODULES:
        mod = _load(name)
        with mock.patch("pathlib.Path.read_text", return_value=broken_text):
            try:
                mod._panichello_directory(Path("/nonexistent"))
            except KeyError:
                pass
            else:
                raise AssertionError(
                    f"{name}._panichello_directory silently tolerated a missing "
                    "'panichello_2024' registry key instead of raising -- the "
                    "silent-default defect is back."
                )
    print("PASS: all 4 loader helpers raise KeyError on a missing registry key.")


def test_confound_table_marks_void_arm_explicitly() -> None:
    mod = _load("run_stimulation_latent_response_map")
    n_by_arm = {"ds005489_openloop": 5, "ds005557_closedloop": 5,
                "haslacher_clam_tacs": 5, "macaque_pfc_microstimulation": None}
    arm_void_reason = {"ds005489_openloop": None, "ds005557_closedloop": None,
                        "haslacher_clam_tacs": None,
                        "macaque_pfc_microstimulation": "reproduction gate voided this arm"}
    rows = mod.confound_table(n_by_arm, arm_void_reason)

    macaque_pfc_microstimulation_row = next(r for r in rows if r["arm"] == "macaque_pfc_microstimulation")
    assert macaque_pfc_microstimulation_row["n"] is None
    assert macaque_pfc_microstimulation_row["n_status"] == "not_computed_this_run_reproduction_gate_void", macaque_pfc_microstimulation_row
    assert macaque_pfc_microstimulation_row["n_status_reason"] == "reproduction gate voided this arm"

    computed_row = next(r for r in rows if r["arm"] == "ds005489_openloop")
    assert computed_row["n"] == 5
    assert computed_row["n_status"] == "computed"
    assert "n_status_reason" not in computed_row

    # a missing key in either dict must raise, not silently produce another None
    try:
        mod.confound_table({k: v for k, v in n_by_arm.items() if k != "macaque_pfc_microstimulation"}, arm_void_reason)
    except KeyError:
        pass
    else:
        raise AssertionError("confound_table tolerated a missing arm key in n_by_arm instead of raising")
    print("PASS: confound_table distinguishes a void arm's null from a measured value, and raises on a missing arm key.")


if __name__ == "__main__":
    test_helpers_raise_on_missing_registry_key()
    test_confound_table_marks_void_arm_explicitly()
    print("ALL CHECKS PASSED")
