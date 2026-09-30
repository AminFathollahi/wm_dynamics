import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from corpus_sessions import MIN_TRIALS, independent_unit, iter_dandi_000004  # noqa: E402

EXPECTED_ENTRY_KEYS = {
    "dataset", "patient", "session", "structure", "spike_lists", "epoch_onsets", "epoch_windows",
    "memorandum_content", "memorandum_content_field", "accuracy", "accuracy_field", "response_time",
}


def _write_session(
    path: Path,
    n_learn: int = 10,
    n_recog: int = 25,
    n_units_per_region: int = 8,
    spiking_units_per_region: int | None = None,
) -> None:
    if spiking_units_per_region is None:
        spiking_units_per_region = n_units_per_region
    n_trials = n_learn + n_recog
    stim_on = np.arange(n_trials, dtype=float) * 10.0
    stim_off = stim_on + 1.0
    delay1 = stim_off + 0.5
    delay2 = delay1 + 2.0
    response_time = delay1 + 0.8
    phase = np.array([b"learn"] * n_learn + [b"recog"] * n_recog)
    labels = np.array([b"NA"] * n_learn + [(b"0" if i % 2 == 0 else b"1") for i in range(n_recog)])
    learn_response = np.zeros(n_learn, dtype=float)
    recog_response = np.array([34.0 if i % 2 == 0 else 31.0 for i in range(n_recog)])
    response_value = np.concatenate([learn_response, recog_response])
    stim_category = (np.arange(n_trials) % 5) + 1

    locations = (
        [b"Left Hippocampus"] * n_units_per_region + [b"Right Amygdala"] * n_units_per_region
    )
    n_units = len(locations)
    electrodes = np.arange(n_units)
    spike_lists = []
    for region_index, count in enumerate((n_units_per_region, n_units_per_region)):
        for unit_index in range(count):
            if unit_index < spiking_units_per_region:
                onsets = delay1[n_learn:]
                spike_lists.append(np.sort(np.concatenate([onsets + 0.2, onsets + 0.4])))
            else:
                spike_lists.append(np.array([]))
    spike_times_flat = np.concatenate(spike_lists) if spike_lists else np.array([])
    spike_times_index = np.cumsum([len(s) for s in spike_lists])

    with h5py.File(path, "w") as handle:
        handle.create_dataset("identifier", data=path.stem.encode())
        electrodes_group = handle.create_group("general/extracellular_ephys/electrodes")
        electrodes_group.create_dataset("location", data=np.array(locations))
        units = handle.create_group("units")
        units.create_dataset("id", data=np.arange(n_units))
        units.create_dataset("electrodes", data=electrodes)
        units.create_dataset("spike_times", data=spike_times_flat)
        units.create_dataset("spike_times_index", data=spike_times_index)
        trials = handle.create_group("intervals/trials")
        trials.create_dataset("id", data=np.arange(n_trials))
        trials.create_dataset("stim_phase", data=phase)
        trials.create_dataset("new_old_labels_recog", data=labels)
        trials.create_dataset("response_value", data=response_value)
        trials.create_dataset("response_time", data=response_time)
        trials.create_dataset("delay1_time", data=delay1)
        trials.create_dataset("delay2_time", data=delay2)
        trials.create_dataset("stimCategory", data=stim_category)


def _entries(root: Path) -> list[dict]:
    return list(iter_dandi_000004(root))


def test_admits_only_recognition_phase_trials_with_valid_labels(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-one_ses-a.nwb", n_learn=10, n_recog=25)

    entries = _entries(tmp_path)
    assert entries
    for entry in entries:
        assert len(entry["memorandum_content"]) == 25
        assert len(entry["accuracy"]) == 25
        assert len(entry["response_time"]) == 25
        assert len(entry["epoch_onsets"]["delay"]) == 25


def test_regions_are_never_pooled(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-one_ses-a.nwb")

    entries = _entries(tmp_path)
    structures = {entry["structure"] for entry in entries}
    assert structures == {"hippocampus", "amygdala"}
    assert "pooled" not in structures


def test_entry_shape_regions_and_resampling_unit_are_patient_clustered(tmp_path):
    subject = tmp_path / "000004" / "sub-p1"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-p1_ses-a.nwb")

    entries = _entries(tmp_path)
    assert entries

    structures = {entry["structure"] for entry in entries}
    assert structures == {"hippocampus", "amygdala"}
    assert len(entries) == 2

    for entry in entries:
        assert EXPECTED_ENTRY_KEYS.issubset(entry)
        session_id = f"{entry['patient']}__{entry['session']}"
        resampling_unit = independent_unit("dandi_000004_human_hippocampus", session_id)
        assert resampling_unit == entry["patient"] == "sub-p1"
        assert resampling_unit != entry["session"]


def test_accuracy_reuses_recognition_correct_semantics(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-one_ses-a.nwb")

    entry = _entries(tmp_path)[0]
    labels = np.array(["0" if i % 2 == 0 else "1" for i in range(25)])
    responses = np.array([34.0 if i % 2 == 0 else 31.0 for i in range(25)])
    from corpus_sessions import recognition_correct

    expected = recognition_correct(labels, responses)
    np.testing.assert_array_equal(entry["accuracy"], expected)
    assert entry["accuracy"].all()


def test_region_below_unit_floor_is_dropped_but_sibling_region_still_yields(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-one_ses-a.nwb", spiking_units_per_region=3)

    entries = _entries(tmp_path)
    assert entries == []


def test_missing_required_trial_column_skips_file(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    path = subject / "sub-one_ses-a.nwb"
    _write_session(path)
    with h5py.File(path, "a") as handle:
        del handle["intervals/trials/stimCategory"]

    assert _entries(tmp_path) == []


def test_below_min_trial_floor_skips_session(tmp_path):
    subject = tmp_path / "000004" / "sub-one"
    subject.mkdir(parents=True)
    _write_session(subject / "sub-one_ses-a.nwb", n_learn=2, n_recog=MIN_TRIALS - 1)

    assert _entries(tmp_path) == []
