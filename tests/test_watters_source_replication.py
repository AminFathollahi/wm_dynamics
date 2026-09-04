import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_watters_source_replication.py"
SPEC = importlib.util.spec_from_file_location("run_watters_source_replication", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_source_bootstrap_probability_prefers_dominant_likelihood():
    likelihood = np.vstack((np.ones(200), np.zeros(200), -np.ones(200)))
    probability = MODULE.source_bootstrap_probabilities(
        likelihood, np.random.default_rng(4), n_boot=50
    )
    assert probability.shape == (50, 3)
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert float(probability[:, 0].mean()) > 0.999


def test_behavior_correctness_uses_target_index():
    frame = pd.DataFrame({
        "completed": [True, True],
        "target_object_index": [0, 1],
        "object_0_r": [1.0, 1.0],
        "object_0_theta": [0.0, 0.0],
        "object_1_r": [1.0, 1.0],
        "object_1_theta": [np.pi, np.pi],
        "response_r": [1.0, 1.0],
        "response_theta": [0.0, 0.0],
        "response_time": [2.0, 2.0],
        "time_cue_onset": [1.0, 1.0],
    })
    result = MODULE.add_behavior_columns(frame, "ring")
    assert result.correct.tolist() == [True, False]
    assert result.reaction_time_ms.tolist() == [1000.0, 1000.0]
