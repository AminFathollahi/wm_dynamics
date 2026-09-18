from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ds006848_feasibility import assess, write_result


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def add_recording(dataset: Path, subject: str, task: str, with_data: bool = True) -> None:
    stem = dataset / subject / "eeg" / f"{subject}_task-{task}"
    write(stem.with_name(stem.name + "_eeg.vhdr"),
          f"DataFile={subject}_task-{task}_eeg.eeg\nMarkerFile={subject}_task-{task}_eeg.vmrk\n")
    write(stem.with_name(stem.name + "_eeg.vmrk"), "Brain Vision Data Exchange Marker File")
    if with_data:
        write(stem.with_name(stem.name + "_eeg.eeg"), "not read")
    else:
        write(stem.with_name(stem.name + "_eeg.eeg.partial"), "partial")
    write(stem.with_name(stem.name + "_eeg.json"), json.dumps({
        "SamplingFrequency": 1000,
        "EEGChannelCount": 2,
        "ECGChannelCount": 1,
        "MiscChannelCount": 1,
        "RecordingType": "continuous",
    }))
    write(stem.with_name(stem.name + "_channels.tsv"),
          "name\ttype\tstatus\nFz\tEEG\tgood\nCz\tEEG\tgood\nECG\tECG\tgood\nPPG\tMISC\tgood\n")


def add_task_tables(dataset: Path, subject: str) -> None:
    write(dataset / subject / "beh" / f"{subject}_task-verbalwm_beh.tsv",
          "participant_id\tevent_id\tcondition\ttrial\tsequence\tresponse\tNCorrect\tpartialScore\ttriggerCorrect\n"
          f"{subject}\t40\tFast\t1\t1234567\t1234567\t7\t7\t1111111\n"
          f"{subject}\t28\tSimultaneous\t1\t7654321\t7654321\t7\t7\t1111111\n")
    events = (
        "onset\tduration\ttrial_type\tvalue\tsample\n"
        "0\t0\tBaseline_2s\t102\t0\n"
        "2\t0\tEncoding_DigitValue_1\t1\t2000\n"
        "2.4\t0\tEncoding_DigitValue_2\t2\t2400\n"
        "2.8\t0\tEncoding_DigitValue_3\t3\t2800\n"
        "3.2\t0\tEncoding_DigitValue_4\t4\t3200\n"
        "3.6\t0\tEncoding_DigitValue_5\t5\t3600\n"
        "4\t0\tEncoding_DigitValue_6\t6\t4000\n"
        "4.4\t0\tEncoding_DigitValue_7\t7\t4400\n"
        "4.8\t0\tRetention_Fast\t40\t4800\n"
        "10.8\t0\tDigits_Retrieval\t110\t10800\n"
        "12\t0\tBaseline_2s\t102\t12000\n"
        "14.8\t0\tEncoding_Set_Simultaneous\t11\t14800\n"
        "14.8\t0\tRetention_Simultaneous\t28\t14800\n"
        "20.8\t0\tDigits_Retrieval\t110\t20800\n"
    )
    write(dataset / subject / "eeg" / f"{subject}_task-verbalwm_events.tsv", events)


def fixture_dataset(root: Path) -> Path:
    dataset = root / "ds006848"
    write(dataset / "dataset_description.json", json.dumps({
        "Name": "AlphaDirection1",
        "DatasetDOI": "doi:10.18112/openneuro.ds006848.v1.0.0",
        "License": "CC0",
        "BIDSVersion": "1.7.0",
        "DatasetType": "raw",
    }))
    write(dataset / ".gitattributes", "* annex.backend=SHA256E\n*.json annex.largefiles=largerthan=1mb\n")
    (dataset / ".datalad").mkdir()
    write(dataset / "participants.tsv",
          "participant_id\tEEG_excluded\tRS_excluded\tbehavior_excluded\n"
          "sub-001\tno\tno\tno\nsub-002\tno\tyes\tno\n")
    for subject in ("sub-001", "sub-002"):
        add_recording(dataset, subject, "verbalwm", with_data=subject == "sub-001")
        add_task_tables(dataset, subject)
    add_recording(dataset, "sub-001", "rest")
    write(dataset / "sub-001" / "eeg" / "sub-001_task-rest_events.tsv",
          "onset\tduration\ttrial_type\tvalue\tsample\n0\t0\tEyes_Open\t10\t0\n")
    return dataset


def test_assess_reports_unavailable_root(tmp_path):
    payload = assess(tmp_path / "missing")
    assert payload["status"] == "not_inspectable"


def test_assess_audits_metadata_without_opening_signal_payloads(tmp_path):
    fixture_dataset(tmp_path)

    payload = assess(tmp_path)

    assert payload["status"] == "complete"
    assert payload["participants"]["n_rows"] == 2
    assert payload["recording_coverage"]["rest_exclusion_matches_missing"] is True
    assert payload["payload_availability"]["data_states"] == {"partial": 1, "regular_present": 2}
    assert payload["payload_availability"]["annex_queryable"] is False
    assert payload["channels"]["ecg_names"] == ["ECG"]
    assert payload["channels"]["ppg_names"] == ["PPG"]
    assert payload["channels"]["ppg_bids_types"] == ["MISC"]
    assert payload["behavior"]["outcome_missing_counts"] == {
        "NCorrect": 0, "partialScore": 0, "response": 0, "triggerCorrect": 0,
    }
    assert payload["events"]["retention_to_retrieval_seconds"]["n_near_six_seconds"] == 4
    assert payload["trial_outcome_linkage"]["sequential_sequences_matched_to_behavior"]["Fast"] == 2
    assert payload["trial_outcome_linkage"]["n_subjects_with_exact_condition_counts"] == 2
    assert payload["feasibility"]["working_memory_autonomic_control"]["status"] == "partially_staged"
    assert payload["feasibility"]["working_memory_autonomic_control"]["n_payload_ready_subjects"] == 1
    assert payload["feasibility"]["working_memory_autonomic_control"]["n_linkage_ready_subjects"] == 1
    assert payload["feasibility"]["working_memory_autonomic_control"]["linkage_ready_subjects"] == ["sub-001"]


def test_write_result_replaces_atomically(tmp_path):
    output = tmp_path / "audit.json"
    write_result(output, {"status": "first"})
    write_result(output, {"status": "second"})
    assert json.loads(output.read_text()) == {"status": "second"}
    assert not output.with_suffix(".json.tmp").exists()
