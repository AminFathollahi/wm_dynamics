import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_watters_item_count_drift.py"
SPEC = importlib.util.spec_from_file_location("run_watters_item_count_drift", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aggregate_delay_bins_is_unsmoothed_and_nonoverlapping():
    values = np.arange(300, dtype=float)
    result = MODULE.aggregate_delay_bins(values, 100)
    assert result.shape == (10,)
    assert result[0] == np.arange(120, 130).sum()
    assert result[-1] == np.arange(210, 220).sum()


def test_session_summary_recovers_positive_total_diffusion_slope():
    rows = []
    for fold in (0, 1):
        for items in (1, 2, 3):
            rows.append({
                "fold": fold,
                "num_objects": items,
                "status": "complete",
                "total_diffusion_standardized_latent_variance_per_s": float(items),
                "diffusion_per_item": 1.0,
                "effective_diffusive_dimensions": 2.0,
                "identifiable_components": 4,
            })
    result = MODULE.session_summary(rows)
    assert result["status"] == "complete"
    assert np.isclose(result["slope_per_added_item"]["total_diffusion"], 1.0)
    assert abs(result["slope_per_added_item"]["diffusion_per_item"]) < 1e-12
