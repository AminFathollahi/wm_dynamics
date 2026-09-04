import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_switching_adjudication",
    ROOT / "scripts" / "run_switching_adjudication.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_dandi_000574_sessions_share_patient_inferential_unit():
    fold = {
        "n_train": 4,
        "n_test": 2,
        "state_space": {"diagnostics": {"n_time": 5}},
        "switching_two_state": {"status": "complete"},
    }
    artifact = {
        "bin_ms": 50,
        "sessions": {
            "sub-01_ses-01": {"status": "complete", "folds": [fold]},
            "sub-01_ses-02": {"status": "complete", "folds": [fold]},
            "sub-02_ses-01": {"status": "complete", "folds": [fold]},
        },
    }

    rows = MODULE.extract_folds("DANDI 000574", artifact)

    assert [row["entity"] for row in rows] == ["sub-01", "sub-01", "sub-02"]
