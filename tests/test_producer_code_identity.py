import ast
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PRODUCERS = [
    "run_000469_pipeline", "run_001187_pipeline", "run_000673_pipeline",
    "run_000574_units_pipeline", "run_boran_pipeline", "run_miller_ctg_corrected",
    "run_ctg_offdiagonal_temporal_stability", "run_behavior_ctg", "run_dim_robustness",
    "run_contraction_behavior_analysis_000469", "run_context_code_cross_dataset_rsa",
    "run_context_confidence_timecourse", "run_decoder_confidence_timecourse_000469",
    "run_divergence_analysis", "run_multiband_analysis", "run_ondemand_streak_diagnostic",
    "build_ecog_epoch_arrays", "build_ecog_geometry_arrays", "build_ecog_dynamics_arrays",
]
IDENTITY_FUNCTIONS = {"code_identity", "write_code_identity_record"}


def _called_names(tree):
    return {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


@pytest.mark.parametrize("producer", PRODUCERS)
def test_producer_records_code_identity(producer):
    tree = ast.parse((SCRIPTS / f"{producer}.py").read_text())
    assert _called_names(tree) & IDENTITY_FUNCTIONS
