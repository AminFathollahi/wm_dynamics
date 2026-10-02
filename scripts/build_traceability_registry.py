#!/usr/bin/env python3
"""Write provenance/traceability_registry.json: one row per frozen or needs_ruling analysis cell.

Each row links a data-lock entry, a hypothesis, a link and a test to the existing script that computes
the cell's measure, then to the artifact, key path, figure panel and manuscript sentence label that
are filled in as results are delivered. A producer is listed only where the script computes the
cell's primary measure on that corpus; related scripts that compute a neighbouring measure are listed
separately and never count as the producer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    if str(ROOT / _subdir) not in sys.path:
        sys.path.insert(0, str(ROOT / _subdir))

from provenance import canonical_json, sha256_file, write_code_identity_record  # noqa: E402

FREEZE_PATH = ROOT / "preregistration" / "analysis_freeze.json"
LOCK_PATH = ROOT / "provenance" / "data_lock.json"
REGISTRY_PATH = ROOT / "provenance" / "traceability_registry.json"

NOT_WRITTEN = "not_written"
PRODUCER_GAP = ("computes the measure under the earlier test and unit inclusion; the frozen test, covariates, "
                "state spaces and common-scale estimate are not yet implemented in it")

RAM = ("ram_ds005489_openloop", "ram_ds005557_closedloop")
HUMAN_UNITS_WITH_CATEGORIES = ("dandi_000469", "dandi_001187", "dandi_000673")
HUMAN_MAINTENANCE_LINK = ("dandi_000469", "dandi_001187", "dandi_000574")

# (hypothesis, link) -> corpus -> script that computes the cell's primary measure on that corpus.
PRODUCERS = {
    ("H1", 4): {c: "run_selective_persistence_and_attractor_geometry.py" for c in HUMAN_UNITS_WITH_CATEGORIES},
    ("H5", 1): {"macaque_pfc_microstimulation": "run_microstimulation_selectivity_mechanism.py"},
    ("H5", 2): {"macaque_pfc_microstimulation": "run_macaque_pfc_microstimulation_recovery_latency.py",
                **{c: "run_ram_stimulation_state_excursion.py" for c in RAM}},
}

# Scripts that compute a neighbouring measure on the corpus (for example a rate-free deviation component
# instead of the distance to a class attractor) and are the starting point for the cell's producer.
RELATED = {
    ("H2", 6): {**{c: ["run_human_maintenance_behaviour_link.py"] for c in HUMAN_MAINTENANCE_LINK},
                "panichello_2024": ["run_rate_free_state_geometry_behavior_link.py"],
                "macaque_pfc_microstimulation": ["run_macaque_maintenance_behaviour_link.py"]},
    ("H3", 7): {"panichello_2024": ["run_rate_free_state_geometry_behavior_link.py"],
                "macaque_pfc_microstimulation": ["run_behavior_amplitude_rate_controls.py"]},
    ("H3", 5): {"panichello_2024": ["run_behavior_amplitude_rate_controls.py"],
                "macaque_pfc_microstimulation": ["run_behavior_amplitude_rate_controls.py"]},
    ("H4", 6): {**{c: ["run_human_maintenance_behaviour_link.py"] for c in HUMAN_MAINTENANCE_LINK},
                "ds004752": ["run_recording_tier_component_transfer.py"]},
    ("H4", 10): {"ds004752": ["run_recording_tier_component_transfer.py"],
                 "dandi_000574": ["run_recording_tier_component_transfer.py"]},
    ("H5", 2): {"haslacher_clam_tacs": ["run_haslacher_stimulation_geometry.py"],
                "alagapan_phase_stimulation": ["run_alagapan_stimulation_geometry.py"],
                "ds005034": ["run_ds005034_tacs_aftereffect.py"]},
    ("H6", 3): {"ram_ds005489_openloop": ["run_ram_openloop_pipeline.py"],
                "ram_ds005557_closedloop": ["run_ram_closedloop_pipeline.py"],
                "macaque_pfc_microstimulation": ["run_macaque_pfc_microstimulation_design_corrected.py"],
                "haslacher_clam_tacs": ["run_phase_locked_scalp_stimulation_component.py"],
                "alagapan_phase_stimulation": ["run_alagapan_stimulation_geometry.py"]},
    ("H6", 8): {"ram_ds005489_openloop": ["run_ram_randomised_prestimulation_moderation.py"],
                "ram_ds005557_closedloop": ["run_ram_stimulation_state_excursion.py"]},
    ("H7", 10): {"macaque_pfc_microstimulation": ["run_macaque_pfc_microstimulation_site_reproducibility.py"],
                 **{c: ["run_stimulation_site_targeting_map.py"] for c in RAM}},
    ("H7", 11): {c: ["run_stimulation_timing_and_parameter_structure.py"] for c in RAM},
    ("H8", 9): {"macaque_pfc_microstimulation": ["run_macaque_maintenance_behaviour_link.py"]},
    ("H10", 12): {c: ["run_cross_modality_calibration.py", "run_recording_tier_component_transfer.py"]
                  for c in ("dandi_000574", "ds004752")},
}


def producer_for(hypothesis: str, link: int, corpus: str) -> str:
    script = PRODUCERS.get((hypothesis, link), {}).get(corpus)
    if script is None:
        return NOT_WRITTEN
    if not (ROOT / "scripts" / script).is_file():
        raise SystemExit(f"producer script is missing: scripts/{script}")
    return script


def related_for(hypothesis: str, link: int, corpus: str) -> list[str]:
    scripts = RELATED.get((hypothesis, link), {}).get(corpus, [])
    missing = [s for s in scripts if not (ROOT / "scripts" / s).is_file()]
    if missing:
        raise SystemExit(f"related script is missing: {missing}")
    return scripts


def build_rows(freeze: dict, lock: dict) -> list[dict]:
    rows = []
    for cell in freeze["cells"]:
        if cell["status"] not in ("frozen", "needs_ruling"):
            continue
        producer = producer_for(cell["hypothesis"], cell["link"], cell["corpus"])
        rows.append({
            "id": cell["id"],
            "data_lock_entry": {"corpus": cell["corpus"], "lock_status": lock["corpora"][cell["corpus"]]["status"]},
            "hypothesis": cell["hypothesis"],
            "link": cell["link"],
            "freeze_status": cell["status"],
            "test": cell["test"],
            "producer": producer,
            "producer_gap": None if producer == NOT_WRITTEN else PRODUCER_GAP,
            "related_scripts": related_for(cell["hypothesis"], cell["link"], cell["corpus"]),
            "artifact": None,
            "key_path": None,
            "figure_panel": None,
            "manuscript_sentence_label": None,
        })
    return rows


def main() -> None:
    freeze = json.loads(FREEZE_PATH.read_text())
    lock = json.loads(LOCK_PATH.read_text())
    rows = build_rows(freeze, lock)
    registry = {"version": "traceability_registry_v1", "freeze_sha256": sha256_file(FREEZE_PATH),
                "steps": ["data_lock_entry", "hypothesis", "link", "test", "producer", "artifact", "key_path",
                          "figure_panel", "manuscript_sentence_label"],
                "rows": rows}
    REGISTRY_PATH.write_text(canonical_json(registry))
    write_code_identity_record(ROOT, Path(__file__), [str(REGISTRY_PATH.relative_to(ROOT))])
    with_producer = sum(r["producer"] != NOT_WRITTEN for r in rows)
    print(json.dumps({"rows": len(rows), "with_producer": with_producer, "not_written": len(rows) - with_producer}))


if __name__ == "__main__":
    main()
