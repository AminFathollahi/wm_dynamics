import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pytest  # noqa: E402

from data_integrity import _completeness_gate, missing_files  # noqa: E402


def _write(path: Path, content: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _build_watters_tree(root: Path) -> None:
    spikes = root / "Watters" / "data_for_modeling" / "data_for_modeling" / "spikes_per_trial"
    behavior = root / "Watters" / "data_for_figures" / "data_for_figures" / "behavior_processing"
    for variant in ("ring", "triangle"):
        path = behavior / f"{variant}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["subject", "session"])
            if variant == "ring":
                writer.writerow(["monkeyA", "20220101"])

    tier_dir = spikes / "monkeyA" / "20220101" / "probe1" / "good"
    _write(tier_dir / "unit1_trials.pkl")
    _write(tier_dir / "unit1_spike_counts.pkl")
    _write(tier_dir / "unit2_trials.pkl")


def _build_ds006848_tree(root: Path) -> None:
    complete_dir = root / "ds006848" / "sub-01" / "eeg"
    _write(complete_dir / "sub-01_task-verbalwm_eeg.vhdr")
    _write(complete_dir / "sub-01_task-verbalwm_eeg.eeg")
    _write(complete_dir / "sub-01_task-verbalwm_eeg.vmrk")

    broken_dir = root / "ds006848" / "sub-02" / "eeg"
    _write(broken_dir / "sub-02_task-verbalwm_eeg.vhdr")
    _write(broken_dir / "sub-02_task-verbalwm_eeg.vmrk")


def _build_dandidownload_staging(root: Path) -> Path:
    staging = root / "000469" / "sub-01" / "partial.nwb.dandidownload"
    _write(staging)
    return staging


def test_completeness_gate_names_exactly_the_broken_files(tmp_path):
    _build_watters_tree(tmp_path)
    _build_ds006848_tree(tmp_path)
    staging_path = _build_dandidownload_staging(tmp_path)

    missing = _completeness_gate(tmp_path)
    missing_paths = {path for _corpus, path in missing}

    expected_watters = str(
        tmp_path / "Watters" / "data_for_modeling" / "data_for_modeling" / "spikes_per_trial"
        / "monkeyA" / "20220101" / "probe1" / "good" / "unit2_spike_counts.pkl"
    )
    expected_vmrk_ok = str(tmp_path / "ds006848" / "sub-02" / "eeg" / "sub-02_task-verbalwm_eeg.eeg")

    assert len(missing) == 3
    assert expected_watters in missing_paths
    assert expected_vmrk_ok in missing_paths
    assert any(str(staging_path) in path for path in missing_paths)

    corpora = {corpus for corpus, _path in missing}
    assert corpora == {"watters_2026_macaque_multi_object", "ds006848_human_scalp", "dandi_000469_human"}


def test_completeness_gate_clean_tree_reports_nothing(tmp_path):
    _write(tmp_path / "Watters" / "data_for_figures" / "data_for_figures"
           / "behavior_processing" / "ring.csv", b"subject,session\n")
    _write(tmp_path / "Watters" / "data_for_figures" / "data_for_figures"
           / "behavior_processing" / "triangle.csv", b"subject,session\n")
    complete_dir = tmp_path / "ds006848" / "sub-01" / "eeg"
    _write(complete_dir / "sub-01_task-verbalwm_eeg.vhdr")
    _write(complete_dir / "sub-01_task-verbalwm_eeg.eeg")
    _write(complete_dir / "sub-01_task-verbalwm_eeg.vmrk")

    assert _completeness_gate(tmp_path) == []


def _build_ds005034_tree(root: Path) -> Path:
    eeg_dir = root / "ds005034" / "sub-01" / "ses-sham" / "eeg"
    stem = "sub-01_ses-sham_task-memory"
    for suffix in ("_eeg.json", "_eeg.set", "_events.tsv", "_channels.tsv"):
        _write(eeg_dir / f"{stem}{suffix}")

    absent_recording = root / "ds005034" / "sub-02" / "ses-verum" / "eeg"
    stem = "sub-02_ses-verum_task-memory"
    for suffix in ("_eeg.json", "_events.tsv", "_channels.tsv"):
        _write(absent_recording / f"{stem}{suffix}")
    return absent_recording / f"{stem}_eeg.set"


def test_missing_files_reports_a_recording_whose_sidecars_landed_first(tmp_path):
    expected = _build_ds005034_tree(tmp_path)
    assert missing_files(tmp_path, ["ds005034"]) == [("ds005034", str(expected))]


def test_missing_files_raises_on_an_unregistered_corpus(tmp_path):
    with pytest.raises(KeyError):
        missing_files(tmp_path, ["a_corpus_that_was_never_registered"])
