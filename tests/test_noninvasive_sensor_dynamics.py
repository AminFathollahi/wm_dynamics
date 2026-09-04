import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_noninvasive_sensor_dynamics.py"
SPEC = importlib.util.spec_from_file_location("run_noninvasive_sensor_dynamics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wolff_lambda_extracts_only_identifiable_participants():
    participants = [
        {"measures": {"voltage": {"endogenous_state_space": {"status": "identifiable", "lambda_rate": 5.0}}}},
        {"measures": {"voltage": {"endogenous_state_space": {"status": "not_identifiable", "lambda_rate": 9.0}}}},
        {"measures": {"voltage": {"endogenous_state_space": {"status": "identifiable", "lambda_rate": 7.0}}}},
    ]
    values = MODULE._wolff_lambda(participants, "voltage")
    assert sorted(values.tolist()) == [5.0, 7.0]


def test_haslacher_lambda_averages_identifiable_components_per_participant():
    participants = [
        {
            "status": "complete",
            "phase_diffusion": {
                "0": {"state_space_components": [
                    {"status": "identifiable", "lambda_rate": 4.0},
                    {"status": "not_identifiable", "lambda_rate": 99.0},
                ]},
                "60": {"state_space_components": [
                    {"status": "identifiable", "lambda_rate": 6.0},
                ]},
            },
        },
        {"status": "excluded", "phase_diffusion": {}},
    ]
    values = MODULE._haslacher_lambda(participants)
    assert values.tolist() == [5.0]


def test_summarize_reports_non_identified_below_five_participants():
    result = MODULE._summarize("x", np.array([1.0, 2.0]), n_total_participants=10)
    assert result["status"] == "non_identified"


def test_summarize_reports_bootstrap_interval_at_five_or_more():
    result = MODULE._summarize("x", np.full(20, 4.0), n_total_participants=20)
    assert result["status"] == "estimable"
    assert result["lambda_rate_per_second_mean"] == 4.0
    lower, upper = result["lambda_rate_per_second_participant_bootstrap_ci"]
    assert lower <= 4.0 <= upper
