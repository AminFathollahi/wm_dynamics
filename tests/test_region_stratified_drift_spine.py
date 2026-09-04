"""Regression tests for the region-stratification added to the human drift
spine runners: the region-conditional minimum-unit
threshold and non_identified short-circuit, and the deciding within-patient
paired contrast helper. Does not re-test the underlying numerical fit (the
full "complete" path), which is unchanged and already exercised by
tests/test_drift_dynamics.py and production runs -- only the new filtering
and threshold logic layered on top of it.
"""

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_human_drift_spine_000469 as spine469  # noqa: E402
import run_human_drift_spine_000574 as spine574  # noqa: E402
import run_human_drift_spine_001187_000673 as spine1187  # noqa: E402


def _write_repeated_item_nwb(path, n_amygdala, n_hippocampus, n_trials=30):
    """Enough of a 000469-shaped NWB for analyze_session's region filter and
    unit-count gate to run, deliberately too few trials to ever reach the
    expensive fold-fitting code (which requires >= 3 repeated items with
    >= N_SPLITS trials each)."""
    n_units = n_amygdala + n_hippocampus
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "general/extracellular_ephys/electrodes/location",
            data=np.array(["amygdala_left"] * n_amygdala + ["hippocampus_left"] * n_hippocampus, dtype="S"),
        )
        f.create_dataset("units/electrodes", data=np.arange(n_units, dtype="int32"))
        f.create_dataset("units/id", data=np.arange(n_units))
        spike_times = np.sort(rng.uniform(0, 100, size=n_units * 50))
        index = np.cumsum([50] * n_units)
        f.create_dataset("units/spike_times", data=spike_times)
        f.create_dataset("units/spike_times_index", data=index)
        onsets = np.arange(1.0, 1.0 + n_trials * 3.0, 3.0)
        f.create_dataset("intervals/trials/id", data=np.arange(n_trials))
        f.create_dataset("intervals/trials/start_time", data=onsets)
        f.create_dataset("intervals/trials/stop_time", data=onsets + 2.3)
        f.create_dataset("intervals/trials/loads", data=np.ones(n_trials, dtype=int))
        f.create_dataset("intervals/trials/loadsEnc1_PicIDs", data=(np.arange(n_trials) % 2))
        f.create_dataset("intervals/trials/response_accuracy", data=np.ones(n_trials, dtype=bool))
        f.create_dataset("intervals/trials/timestamps_Maintenance", data=onsets)


class TestRegionThresholdShortCircuit000469:
    def test_pooled_uses_all_units_and_region_below_threshold_is_nonidentified(self, tmp_path):
        path = tmp_path / "sub-1_ses-2_ecephys+image.nwb"
        _write_repeated_item_nwb(path, n_amygdala=20, n_hippocampus=3)
        pooled = spine469.analyze_session(path, seed=1, region="pooled")
        hippocampus = spine469.analyze_session(path, seed=1, region="hippocampus")
        assert pooled["status"] != "non_identified"  # 23 units clears MIN_UNITS (15)
        assert hippocampus["status"] == "non_identified"
        assert "region=hippocampus" in hippocampus["reason"]

    def test_region_above_threshold_is_not_short_circuited(self, tmp_path):
        path = tmp_path / "sub-2_ses-2_ecephys+image.nwb"
        _write_repeated_item_nwb(path, n_amygdala=10, n_hippocampus=3)
        amygdala = spine469.analyze_session(path, seed=1, region="amygdala")
        # 10 amygdala units clears MIN_UNITS_PER_REGION (8); this session still
        # fails downstream (too few repeated-item trials), but NOT at the
        # unit-count gate -- proves the region-specific threshold, not the
        # pooled one, governs the region path.
        assert amygdala["status"] != "non_identified"


class TestRegionThresholdShortCircuit001187000673:
    def test_region_below_threshold_is_nonidentified(self, tmp_path):
        path = tmp_path / "sub-1_ses-1_ecephys+image.nwb"
        _write_repeated_item_nwb(path, n_amygdala=20, n_hippocampus=3)
        with h5py.File(path, "a") as f:
            f.create_dataset("intervals/WM_trials/id", data=f["intervals/trials/id"][:])
            f.create_dataset("intervals/WM_trials/loads", data=f["intervals/trials/loads"][:])
            f.create_dataset("intervals/WM_trials/timestamps_Maintenance", data=f["intervals/trials/timestamps_Maintenance"][:])
            f.create_dataset("intervals/WM_trials/response_accuracy", data=f["intervals/trials/response_accuracy"][:])
        hippocampus = spine1187.analyze_unit_session(path, "001187", seed=1, region="hippocampus")
        assert hippocampus["status"] == "non_identified"
        pooled = spine1187.analyze_unit_session(path, "001187", seed=1, region="pooled")
        assert pooled["status"] != "non_identified"


def _write_boran_nwb(path, n_hippocampus, n_amygdala, n_trials=30):
    """Enough of a 000574-shaped (Boran) NWB for analyze_session's region
    filter and unit-count gate to run, deliberately too few set-size levels
    to ever reach the expensive fold-fitting code."""
    n_units = n_hippocampus + n_amygdala
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "general/extracellular_ephys/electrodes/location",
            data=np.array(
                ["Hipp, Left Hippocampus cHipp, caudal hippocampus"] * n_hippocampus
                + ["Amy, Left Amygdala"] * n_amygdala,
                dtype="S",
            ),
        )
        f.create_dataset("units/electrodes", data=np.arange(n_units, dtype="int32"))
        f.create_dataset("units/id", data=np.arange(n_units))
        spike_times = np.sort(rng.uniform(0, 100, size=n_units * 50))
        index = np.cumsum([50] * n_units)
        f.create_dataset("units/spike_times", data=spike_times)
        f.create_dataset("units/spike_times_index", data=index)
        start_time = np.arange(1.0, 1.0 + n_trials * 6.0, 6.0)
        f.create_dataset("intervals/trials/id", data=np.arange(n_trials))
        f.create_dataset("intervals/trials/start_time", data=start_time)
        f.create_dataset("intervals/trials/artifact", data=np.zeros(n_trials, dtype=bool))
        f.create_dataset("intervals/trials/set_size", data=np.full(n_trials, 4, dtype=int))
        f.create_dataset("intervals/trials/correct", data=np.ones(n_trials, dtype=bool))


class TestRegionThresholdShortCircuit000574:
    def test_pooled_uses_all_units_and_region_below_threshold_is_nonidentified(self, tmp_path):
        path = tmp_path / "sub-01_ses-01.nwb"
        _write_boran_nwb(path, n_hippocampus=3, n_amygdala=20)
        pooled = spine574.analyze_session(path, seed=1, region="pooled")
        hippocampus = spine574.analyze_session(path, seed=1, region="hippocampus")
        assert pooled["status"] != "non_identified"  # 23 units clears MIN_UNITS (8)
        assert hippocampus["status"] == "non_identified"
        assert "region=hippocampus" in hippocampus["reason"]

    def test_region_above_threshold_is_not_short_circuited(self, tmp_path):
        path = tmp_path / "sub-02_ses-01.nwb"
        _write_boran_nwb(path, n_hippocampus=10, n_amygdala=2)
        hippocampus = spine574.analyze_session(path, seed=1, region="hippocampus")
        # 10 hippocampus units clears MIN_UNITS_PER_REGION (8); this session
        # still fails downstream (only one set-size level), but NOT at the
        # unit-count gate -- proves the region-specific threshold governs.
        assert hippocampus["status"] != "non_identified"


class TestPairedRegionContrast000574:
    """Unlike 000469/001187's session-keyed paired_region_contrast, 000574
    nests multiple sessions per patient, so this pairs on already
    patient-averaged ``patient_level_metrics``."""

    def test_non_identified_below_two_shared_patients(self):
        regions_group = {
            "hippocampus": {"patient_level_metrics": {"sub-01": {"m": 1.0}}},
            "amygdala": {"patient_level_metrics": {"sub-01": {"m": 0.5}}},
        }
        result = spine574.paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "non_identified"

    def test_estimable_with_enough_shared_patients(self):
        regions_group = {
            "hippocampus": {"patient_level_metrics": {
                f"sub-{i:02d}": {"m": 2.0} for i in range(5)
            }},
            "amygdala": {"patient_level_metrics": {
                f"sub-{i:02d}": {"m": 1.0} for i in range(5)
            }},
        }
        result = spine574.paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "estimable"
        assert result["mean_difference"] == pytest.approx(1.0)
        assert result["direction_a_greater_than_b"] is True


def _write_content_axis_nwb(path, n_amygdala, n_hippocampus, n_trials=30):
    """001187-shaped NWB (WM_trials group) with a 2-class load label, enough
    for analyze_session_content_axis's region filter and unit-count gate to
    run, deliberately too few trials per load to ever reach the expensive
    fold-fitting code (needs >= N_SPLITS=5 trials in EACH of both loads)."""
    n_units = n_amygdala + n_hippocampus
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "general/extracellular_ephys/electrodes/location",
            data=np.array(["amygdala_left"] * n_amygdala + ["hippocampus_left"] * n_hippocampus, dtype="S"),
        )
        f.create_dataset("units/electrodes", data=np.arange(n_units, dtype="int32"))
        f.create_dataset("units/id", data=np.arange(n_units))
        spike_times = np.sort(rng.uniform(0, 100, size=n_units * 50))
        index = np.cumsum([50] * n_units)
        f.create_dataset("units/spike_times", data=spike_times)
        f.create_dataset("units/spike_times_index", data=index)
        onsets = np.arange(1.0, 1.0 + n_trials * 3.0, 3.0)
        f.create_dataset("intervals/WM_trials/id", data=np.arange(n_trials))
        f.create_dataset("intervals/WM_trials/loads", data=np.where(np.arange(n_trials) % 2 == 0, 1, 3))
        f.create_dataset("intervals/WM_trials/response_accuracy", data=np.ones(n_trials, dtype=bool))
        f.create_dataset("intervals/WM_trials/timestamps_Maintenance", data=onsets)


class TestContentAxisRegionThresholdShortCircuit001187000673:
    def test_pooled_uses_all_units_and_region_below_threshold_is_nonidentified(self, tmp_path):
        path = tmp_path / "sub-1_ses-1_ecephys+image.nwb"
        _write_content_axis_nwb(path, n_amygdala=20, n_hippocampus=3)
        pooled = spine1187.analyze_session_content_axis(path, "001187", seed=1, region="pooled")
        hippocampus = spine1187.analyze_session_content_axis(path, "001187", seed=1, region="hippocampus")
        assert pooled["status"] != "non_identified"  # 23 units clears MIN_UNITS (15)
        assert hippocampus["status"] == "non_identified"
        assert "region=hippocampus" in hippocampus["reason"]

    def test_region_above_threshold_is_not_short_circuited(self, tmp_path):
        path = tmp_path / "sub-2_ses-1_ecephys+image.nwb"
        _write_content_axis_nwb(path, n_amygdala=10, n_hippocampus=3, n_trials=30)
        with h5py.File(path, "a") as f:
            # skew loads so one class has < N_SPLITS trials -- enough total
            # trials for firing-rate QC to pass reliably, too few per-class
            # for the downstream load-count gate.
            del f["intervals/WM_trials/loads"]
            f.create_dataset(
                "intervals/WM_trials/loads",
                data=np.where(np.arange(30) < 26, 1, 3),
            )
        amygdala = spine1187.analyze_session_content_axis(path, "001187", seed=1, region="amygdala")
        # 10 amygdala units clears MIN_UNITS_PER_REGION (8); this session
        # still fails downstream (too few trials in the load=3 class), but
        # NOT at the unit-count gate.
        assert amygdala["status"] != "non_identified"
        assert amygdala["status"] == "excluded"
        assert "load levels" in amygdala["reason"]


class TestContentAxisPatientLevelMeans:
    def test_averages_within_patient_and_skips_incomplete(self):
        sessions = {
            "p1_s1": {"patient": "p1", "content_axis_fit": {
                "status": "complete", "summary": {"m": 1.0},
            }},
            "p1_s2": {"patient": "p1", "content_axis_fit": {
                "status": "complete", "summary": {"m": 3.0},
            }},
            "p2_s1": {"patient": "p2", "content_axis_fit": {"status": "excluded"}},
        }
        result = spine1187.content_axis_patient_level_means(sessions, metric_names=("m",))
        assert result["p1"]["m"] == pytest.approx(2.0)
        assert "p2" not in result


class TestContentAxisPairedRegionContrast:
    def test_non_identified_below_two_shared_patients(self):
        regions_group = {
            "hippocampus": {"patient_level_metrics": {"p1": {"m": 1.0}}},
            "amygdala": {"patient_level_metrics": {"p1": {"m": 0.5}}},
        }
        result = spine1187.content_axis_paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "non_identified"

    def test_estimable_with_enough_shared_patients(self):
        regions_group = {
            "hippocampus": {"patient_level_metrics": {f"p{i}": {"m": 2.0} for i in range(5)}},
            "amygdala": {"patient_level_metrics": {f"p{i}": {"m": 1.0} for i in range(5)}},
        }
        result = spine1187.content_axis_paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "estimable"
        assert result["mean_difference"] == pytest.approx(1.0)
        assert result["direction_a_greater_than_b"] is True


class TestPairedRegionContrast:
    def test_non_identified_below_two_shared_patients(self):
        regions_group = {
            "hippocampus": {"sessions": {"sub-1": {"status": "complete", "summary": {"m": 1.0}}}},
            "amygdala": {"sessions": {"sub-1": {"status": "complete", "summary": {"m": 0.5}}}},
        }
        result = spine469.paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "non_identified"

    def test_estimable_with_enough_shared_patients(self):
        regions_group = {
            "hippocampus": {"sessions": {
                f"sub-{i}": {"status": "complete", "summary": {"m": 2.0}} for i in range(5)
            }},
            "amygdala": {"sessions": {
                f"sub-{i}": {"status": "complete", "summary": {"m": 1.0}} for i in range(5)
            }},
        }
        result = spine469.paired_region_contrast(regions_group, "hippocampus", "amygdala", "m")
        assert result["status"] == "estimable"
        assert result["mean_difference"] == pytest.approx(1.0)
        assert result["direction_a_greater_than_b"] is True
