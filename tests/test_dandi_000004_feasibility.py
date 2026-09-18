import importlib.util
from pathlib import Path

import h5py
import numpy as np

from src.spike_pipeline import normalize_region_label


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_dandi_000004_feasibility", ROOT / "scripts" / "run_dandi_000004_feasibility.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _nwb(path: Path, identifier: str, subject: str) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("identifier", data=identifier.encode())
        handle.create_dataset("general/subject/subject_id", data=subject.encode())
        electrodes = handle.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset("location", data=np.array([b"Right Hippocampus", b"Left Amygdala"]))
        units = handle.create_group("units")
        units.create_dataset("id", data=np.array([1, 2]))
        units.create_dataset("electrodes", data=np.array([0, 1]))
        trials = handle.create_group("intervals/trials")
        trials.create_dataset("id", data=np.array([0, 1]))
        trials.create_dataset("stim_on_time", data=np.array([1.0, 10.0]))
        trials.create_dataset("stim_off_time", data=np.array([2.0, 11.0]))
        trials.create_dataset("delay1_time", data=np.array([2.5, 11.5]))
        trials.create_dataset("delay2_time", data=np.array([4.5, 13.5]))
        trials.create_dataset("response_time", data=np.array([4.0, 13.0]))
        trials.create_dataset("new_old_labels_recog", data=np.array([b"new", b"old"]))
        trials.create_dataset("response_value", data=np.array([1, 2]))


def test_hemisphere_prefixed_nwb_labels_reuse_shared_parser():
    assert normalize_region_label("Right Hippocampus", "nwb_hemisphere_prefixed_structure") == ("hippocampus", "right")
    assert normalize_region_label("Left Amygdala", "nwb_hemisphere_prefixed_structure") == ("amygdala", "left")


def test_feasibility_census_keeps_complete_files_and_separates_partial_download(tmp_path):
    _nwb(tmp_path / "sub-one_ses-a_obj-one.nwb", "session-one", "sub-one")
    _nwb(tmp_path / "sub-two_ses-a_obj-two.nwb", "session-two", "sub-two")
    partial = tmp_path / "sub-three_ses-a.nwb.dandidownload"
    partial.mkdir()
    (partial / "file").write_bytes(b"partial")

    result = MODULE.build_feasibility(tmp_path)

    assert result["file_inventory"]["n_complete_nwb_files_discovered"] == 2
    assert result["file_inventory"]["n_complete_nwb_files_inspected"] == 2
    assert result["file_inventory"]["n_complete_nwb_files_dropped"] == 0
    assert result["status"] == "complete"
    assert result["file_inventory"]["n_incomplete_download_directories"] == 1
    assert result["identity_inventory"]["n_unique_participants"] == 2
    assert result["identity_inventory"]["n_unique_nwb_identifiers"] == 2
    assert result["trial_inventory"]["trial_outcome_value_counts"]["new_old_labels_recog"] == {"new": 2, "old": 2}
    assert result["unit_inventory"]["resolved_unit_region_counts"] == {"amygdala": 2, "hippocampus": 2}
    assert result["zero_drop_audit"]["complete_files_excluded"] == 0
    assert result["files"][0]["named_inter_event_intervals_seconds"]["delay2_minus_delay1"]["min"] == 2.0
    assert result["files"][0]["named_inter_event_intervals_seconds"]["delay2_minus_response"]["min"] == 0.5
