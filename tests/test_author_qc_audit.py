from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_author_qc_audit_covers_every_in_scope_dataset_and_never_auto_eligibilizes_ram():
    # Writes to a temporary path, not results/author_preprocessing_qc_audit.json -- running the test
    # suite must not rewrite a delivered artifact as a side effect (see build_author_qc_audit.py's
    # optional output-path override, added for exactly this).
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_output = Path(tmp_dir) / "author_preprocessing_qc_audit.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_author_qc_audit.py"), str(tmp_output)],
                       check=True, cwd=ROOT, capture_output=True, text=True)
        audit = json.loads(tmp_output.read_text())
    datasets = {row["dataset"]: row for row in audit["datasets"]}
    required_tokens = ("000469", "001187", "000574", "Boran", "Miller", "Wolff", "ALM",
                       "microstimulation", "RAM", "Haslacher", "Alagapan", "TES1", "Panichello", "Watters")
    assert all(any(token in name for name in datasets) for token in required_tokens)
    ram = next(row for name, row in datasets.items() if "RAM" in name)
    if ram["artifact_present"]:
        assert ram["status"] == "eligible_design_gated_encoding_only"
    else:
        assert ram["status"] == "pending_author_feature_and_design_repair"
    assert ram["implemented"]
    assert all(row["required_qc"] for row in datasets.values())
