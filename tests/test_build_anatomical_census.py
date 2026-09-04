"""Regression tests for scripts/build_anatomical_census.py.

Builds a synthetic NWB file (Rutishauser-suffix + Boran-hybrid label
conventions) and a synthetic BIDS electrodes.tsv (RAM ind.region convention)
and checks the per-site census rows and roll-up matrix they produce, without
touching any staged dataset.
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from build_anatomical_census import (  # noqa: E402
    _census_nwb_electrodes,
    _census_ram,
    _coarse_group,
    _roll_up,
)


def _write_synthetic_nwb(path, boran: bool):
    with h5py.File(path, "w") as f:
        if boran:
            locations = [
                "Hipp, Left Hippocampus cHipp, caudal hippocampus",
                "unspecific",
                "F3",
            ]
            group_names = ["ieeg", "ieeg", "eeg"]
            labels = ["AHL1", "AHL2", "F3"]
            unit_electrodes = [0, 0]  # two units on electrode 0, none on 1 (unspecific) or 2 (scalp)
        else:
            locations = ["hippocampus_left", "amygdala_right"]
            group_names = ["ieeg", "ieeg"]
            labels = ["A1", "A2"]
            unit_electrodes = [0, 1]

        e = f.create_group("general/extracellular_ephys/electrodes")
        e.create_dataset("id", data=np.arange(len(locations)))
        e.create_dataset("location", data=np.array(locations, dtype="S"))
        e.create_dataset("label", data=np.array(labels, dtype="S"))
        e.create_dataset("group_name", data=np.array(group_names, dtype="S"))
        e.create_dataset("x", data=np.arange(len(locations), dtype=float))
        e.create_dataset("y", data=np.arange(len(locations), dtype=float))
        e.create_dataset("z", data=np.arange(len(locations), dtype=float))

        f.create_dataset("units/id", data=np.arange(len(unit_electrodes)))
        f.create_dataset("units/electrodes", data=np.array(unit_electrodes, dtype="int32"))


class TestCensusNWBRutishauserFamily:
    def test_sites_and_unit_counts(self, tmp_path):
        path = tmp_path / "sub-1_ses-1.nwb"
        _write_synthetic_nwb(path, boran=False)
        sites = _census_nwb_electrodes(path, "dandi_000469", boran=False)
        assert len(sites) == 2
        assert sites[0]["normalized_structure"] == "hippocampus"
        assert sites[0]["hemisphere"] == "left"
        assert sites[0]["modality"] == "single_unit"
        assert sites[0]["n_units"] == 1
        assert sites[1]["normalized_structure"] == "amygdala"
        assert sites[1]["is_stimulation_site"] is False


class TestCensusNWBBoran:
    def test_scalp_depth_and_unspecific_distinguished(self, tmp_path):
        path = tmp_path / "sub-01_ses-01.nwb"
        _write_synthetic_nwb(path, boran=True)
        sites = _census_nwb_electrodes(path, "dandi_000574", boran=True)
        assert len(sites) == 3
        hipp, unspecific, scalp = sites
        assert hipp["normalized_structure"] == "hippocampus"
        assert hipp["modality"] == "single_unit"
        assert hipp["n_units"] == 2
        assert unspecific["normalized_structure"] == "unspecific"
        assert unspecific["modality"] == "depth_lfp"
        assert unspecific["n_units"] == 0
        assert scalp["normalized_structure"] == "scalp"
        assert scalp["modality"] == "scalp_eeg"


class TestCensusRAMElectrodesTsv:
    def _write_bids_session(self, root, sub, ses, rows, stim_channels=()):
        ieeg = root / sub / ses / "ieeg"
        ieeg.mkdir(parents=True)
        elec_path = ieeg / f"{sub}_{ses}_task-FR2_electrodes.tsv"
        with open(elec_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        if stim_channels:
            ev_path = ieeg / f"{sub}_{ses}_task-FR2_events.tsv"
            with open(ev_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["anode_label", "cathode_label"], delimiter="\t")
                w.writeheader()
                for a, c in stim_channels:
                    w.writerow({"anode_label": a, "cathode_label": c})
        return elec_path

    def test_hand_annotation_preferred_over_ind_region_and_stim_flag(self, tmp_path):
        root = tmp_path / "ds005489-download"
        rows = [
            {
                "name": "LA1", "x": "1.0", "y": "2.0", "z": "3.0",
                "hemisphere": "L", "stein.region": "Left Amy",
                "das.region": "n/a", "ind.region": "n/a",
            },
            {
                "name": "LB1", "x": "4.0", "y": "5.0", "z": "6.0",
                "hemisphere": "L", "stein.region": "n/a",
                "das.region": "n/a", "ind.region": "temporalpole",
            },
            {
                "name": "LC1", "x": "7.0", "y": "8.0", "z": "9.0",
                "hemisphere": "R", "stein.region": "n/a",
                "das.region": "n/a", "ind.region": "totallyunknownlabel",
            },
        ]
        self._write_bids_session(root, "sub-R1026D", "ses-0", rows, stim_channels=[("LC1", "LC2")])
        out = _census_ram("ram_ds005489_openloop", root)
        sites = out["patients"]["sub-R1026D"]["sessions"]["ses-0"]["recording_sites"]
        assert sites[0]["normalized_structure"] == "amygdala"
        assert sites[0]["source_field"].endswith("stein.region")
        assert sites[1]["normalized_structure"] == "temporal_pole"
        assert sites[2]["normalized_structure"] == "other"
        assert sites[2]["raw_label"] == "totallyunknownlabel"  # never silently absorbed
        assert sites[2]["is_stimulation_site"] is True
        assert sites[0]["is_stimulation_site"] is False


class TestRollUp:
    def test_structure_by_dataset_matrix_counts_patients_and_units(self):
        datasets = {
            "dandi_000469": {
                "patients": {
                    "sub-1": {
                        "sessions": {
                            "ses-1": {
                                "recording_sites": [
                                    {"normalized_structure": "hippocampus", "n_units": 3},
                                    {"normalized_structure": "hippocampus", "n_units": 2},
                                    {"normalized_structure": "amygdala", "n_units": 1},
                                ]
                            }
                        }
                    },
                    "sub-2": {
                        "sessions": {
                            "ses-1": {
                                "recording_sites": [
                                    {"normalized_structure": "hippocampus", "n_units": 0},
                                ]
                            }
                        }
                    },
                }
            }
        }
        matrix = _roll_up(datasets)
        assert matrix["hippocampus"]["dandi_000469"]["n_patients"] == 2
        assert matrix["hippocampus"]["dandi_000469"]["n_sites"] == 3
        assert matrix["hippocampus"]["dandi_000469"]["n_units"] == 5
        assert matrix["amygdala"]["dandi_000469"]["n_patients"] == 1


class TestCoarseGroup:
    def test_mtl_structures(self):
        assert _coarse_group("hippocampus") == "MTL"
        assert _coarse_group("vtc") == "MTL"

    def test_declared_single_structure_by_design(self):
        assert _coarse_group("prefrontal_cortex") == "lateral_frontal"
        assert _coarse_group("anterior_lateral_motor_cortex") == "lateral_frontal"

    def test_unknown_structure_is_other(self):
        assert _coarse_group("some_future_structure") == "other"
