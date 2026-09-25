from __future__ import annotations

import json
from pathlib import Path

from corpus_sessions import (
    WATTERS_QUALITY_TIERS, alm_data_directory, pfc4_data_directory, watters_directories,
    watters_session_dates,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_CONFIG_PATH = REPO_ROOT / "config" / "datasets.json"
CANONICAL_RECORDING_REGISTRY_PATH = REPO_ROOT / "provenance" / "canonical_recording_registry.json"
LOCAL_PATH_OVERRIDE = {"dandi_000673": "000673"}

MACAQUE_USTIM_SESSIONS = [
    "Sa210311_s224",
    "Wa220801_s549", "Wa220802_s550", "Wa220803_s551", "Wa220804_s552",
    "Wa220805_s553", "Wa220808_s554", "Wa220809_s555", "Wa220810_s556",
    "Wa220811_s557", "Wa220812_s558",
]
ALAGAPAN_PATIENTS = ("P1", "P2", "P3")
MILLER_SUBJECTS = ("al", "ca", "cc", "ug")
HASLACHER_ACTIVE_SUBJECTS = [f"PA{i}" for i in range(1, 17)] + ["PA18", "PA19", "PA20", "PA22", "PA23"]
HASLACHER_CONTROL_SUBJECTS = ["PA17", "PA21"] + [f"PA{i}" for i in range(24, 47)]
HASLACHER_SUBJECTS = HASLACHER_ACTIVE_SUBJECTS + HASLACHER_CONTROL_SUBJECTS


def _absent(path: Path) -> bool:
    return not path.is_file() or path.stat().st_size == 0


def load_datasets_config() -> dict:
    return json.loads(DATASETS_CONFIG_PATH.read_text())


def _completeness_gate(root: Path) -> list[tuple[str, str]]:
    missing: list[tuple[str, str]] = []

    for corpus, directory in (
        ("dandi_000469_human", root / "000469"), ("dandi_001187_human", root / "001187"),
        ("dandi_000574_human", root / "000574"), ("dandi_000004_human", root / "000004"),
    ):
        if directory.is_dir():
            for staging in sorted(directory.glob("**/*.dandidownload")):
                missing.append((corpus, f"{staging} (interrupted download, target file not yet complete)"))

    spikes_dir, _ = watters_directories(root)
    for animal, date, _variant in watters_session_dates(root):
        session_dir = spikes_dir / animal / date
        if not session_dir.is_dir():
            continue
        for probe_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            for quality in WATTERS_QUALITY_TIERS:
                tier_dir = probe_dir / quality
                if not tier_dir.is_dir():
                    continue
                for trials_path in sorted(tier_dir.glob("*_trials.pkl")):
                    unit_id = trials_path.name[: -len("_trials.pkl")]
                    counts_path = tier_dir / f"{unit_id}_spike_counts.pkl"
                    if _absent(counts_path):
                        missing.append(("watters_2026_macaque_multi_object", str(counts_path)))

    ds006848_dir = root / "ds006848"
    for subject_dir in sorted(ds006848_dir.glob("sub-*")) if ds006848_dir.is_dir() else []:
        vhdr_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.vhdr"
        if not vhdr_path.is_file():
            continue
        for suffix in ("eeg", "vmrk"):
            companion = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.{suffix}"
            if _absent(companion):
                missing.append(("ds006848_human_scalp", str(companion)))

    return missing


def _dataset_local_path(dataset_key: str, config: dict) -> str | None:
    entry = config["datasets"].get(dataset_key, {})
    return entry.get("local_path") or LOCAL_PATH_OVERRIDE.get(dataset_key)


def _dataset_root(root: Path, dataset_key: str, config: dict) -> Path | None:
    local_path = _dataset_local_path(dataset_key, config)
    return root / local_path if local_path else None


def _split(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    present, absent = [], []
    for path in paths:
        (absent if _absent(path) else present).append(path)
    return present, absent


def _glob_two_level(directory: Path | None, subject_glob: str, file_glob: str):
    if directory is None or not directory.is_dir():
        return [], []
    paths = [p for subject_dir in sorted(directory.glob(subject_glob)) for p in sorted(subject_dir.glob(file_glob))]
    return _split(paths)


def _glob_one_level(directory: Path | None, file_glob: str):
    if directory is None or not directory.is_dir():
        return [], []
    return _split(sorted(directory.glob(file_glob)))


def _check_dandi_000469(root: Path, config: dict):
    present, absent = _glob_two_level(_dataset_root(root, "dandi_000469", config), "sub-*", "*_ses-2_ecephys+image.nwb")
    return present, absent, "glob_fail_safe"


def _check_dandi_000574(root: Path, config: dict):
    present, absent = _glob_two_level(_dataset_root(root, "dandi_000574", config), "sub-*", "*.nwb")
    return present, absent, "glob_fail_safe"


def _check_dandi_000004(root: Path, config: dict):
    present, absent = _glob_two_level(_dataset_root(root, "dandi_000004", config), "sub-*", "*.nwb")
    return present, absent, "glob_fail_safe"


def _check_dandi_001187(root: Path, config: dict):
    from run_human_drift_spine_001187_000673 import canonical_sessions

    paths = [root / meta["primary_path"] for meta in canonical_sessions() if meta["primary_release"] == "001187"]
    present, absent = _split(paths)
    return present, absent, "manifest"


def _check_dandi_000673(root: Path, config: dict):
    if not CANONICAL_RECORDING_REGISTRY_PATH.is_file():
        return [], [], "manifest"
    registry = json.loads(CANONICAL_RECORDING_REGISTRY_PATH.read_text())
    paths = [root / row["path"] for row in registry if row.get("release") == "000673"]
    present, absent = _split(paths)
    return present, absent, "manifest"


def _check_panichello_2024(root: Path, config: dict):
    present, absent = _glob_one_level(_dataset_root(root, "panichello_2024", config), "*.mat")
    return present, absent, "glob_fail_safe"


def _check_watters_2026(root: Path, config: dict):
    spikes_dir, _ = watters_directories(root)
    present, absent = [], []
    for animal, date, _variant in watters_session_dates(root):
        session_dir = spikes_dir / animal / date
        if not session_dir.is_dir():
            continue
        for probe_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            for quality in WATTERS_QUALITY_TIERS:
                tier_dir = probe_dir / quality
                if not tier_dir.is_dir():
                    continue
                for trials_path in sorted(tier_dir.glob("*_trials.pkl")):
                    unit_id = trials_path.name[: -len("_trials.pkl")]
                    counts_path = tier_dir / f"{unit_id}_spike_counts.pkl"
                    (absent if _absent(trials_path) else present).append(trials_path)
                    (absent if _absent(counts_path) else present).append(counts_path)
    return present, absent, "explicit"


def _check_inagaki_alm5(root: Path, config: dict):
    present, absent = _glob_one_level(alm_data_directory(root), "*.mat")
    return present, absent, "glob_fail_safe"


def _check_pfc4(root: Path, config: dict):
    directory = pfc4_data_directory(root)
    paths = []
    if directory.is_dir():
        for animal_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
            paths.extend(sorted(animal_dir.rglob("*.mat")))
    present, absent = _split(paths)
    return present, absent, "glob_fail_safe"


def _check_pfc3(root: Path, config: dict):
    present, absent = _glob_one_level(_dataset_root(root, "pfc3", config), "*.mat")
    return present, absent, "glob_fail_safe"


def _check_wolff_eeg_impulse(root: Path, config: dict):
    directory = _dataset_root(root, "wolff_eeg_impulse", config)
    paths = []
    if directory is not None and directory.is_dir():
        paths.extend(sorted(directory.glob("Dynamic_hidden_states_exp1_*.mat")))
        paths.extend(sorted(directory.glob("Dynamic_hidden_states_exp2_*.mat")))
    present, absent = _split(paths)
    return present, absent, "glob_fail_safe"


def _check_ram_bids_stim(dataset_key: str):
    def check(root: Path, config: dict):
        present, absent = _glob_two_level(_dataset_root(root, dataset_key, config), "sub-*",
                                           "ses-*/ieeg/*_acq-bipolar_ieeg.json")
        return present, absent, "glob_fail_safe"
    return check


def _check_ds004752(root: Path, config: dict):
    directory = _dataset_root(root, "ds004752", config)
    paths = []
    if directory is not None and directory.is_dir():
        for patient_dir in sorted(directory.glob("sub-*")):
            for ieeg_dir in sorted(patient_dir.glob("ses-*/ieeg")):
                paths.extend(sorted(ieeg_dir.glob("*_events.tsv")))
    present, absent = _split(paths)
    return present, absent, "glob_fail_safe"


def _check_macaque_pfc_microstimulation(root: Path, config: dict):
    directory = _dataset_root(root, "macaque_pfc_microstimulation", config)
    paths = []
    if directory is not None:
        for prefix in MACAQUE_USTIM_SESSIONS:
            paths.append(directory / "correct" / f"{prefix}.mat")
            paths.append(directory / "error" / f"{prefix}_err.mat")
            paths.append(directory / "shortedchan" / f"{prefix}_shorted.mat")
    present, absent = _split(paths)
    return present, absent, "explicit"


def _check_alagapan_phase_stimulation(root: Path, config: dict):
    directory = _dataset_root(root, "alagapan_phase_stimulation", config)
    present, absent = [], []
    if directory is not None:
        for patient in ALAGAPAN_PATIENTS:
            mapping_paths = (
                directory / "Electrode Information" / "Electrode Mapping" / f"{patient}_ElectrodeMapping_Baseline.csv",
                directory / "Electrode Information" / "Electrode Mapping" / f"{patient}_ElectrodeMapping.csv",
            )
            chosen = next((p for p in mapping_paths if not _absent(p)), None)
            (present if chosen is not None else absent).append(chosen or mapping_paths[0])
            explicit = [
                directory / "Electrode Information" / "Seizure Electrodes" / f"{patient}_seizureElectrodes.mat",
                directory / "Task Performance" / f"{patient}_SternbergStimulation_Summary.csv",
                directory / "iEEG Data" / "Baseline" / f"{patient}_SMS_Baseline_Epoched.set",
                directory / "iEEG Data" / "Baseline" / f"{patient}_SMS_Baseline_Epoched.fdt",
                directory / "iEEG Data" / "Stimulation" / f"{patient}_SMS_Stimulation.set",
                directory / "iEEG Data" / "Stimulation" / f"{patient}_SMS_Stimulation.fdt",
            ]
            for path in explicit:
                (absent if _absent(path) else present).append(path)
    return present, absent, "explicit"


def _check_haslacher_clam_tacs(root: Path, config: dict):
    directory = _dataset_root(root, "haslacher_clam_tacs", config)
    present, absent = [], []
    if directory is not None:
        for subject in HASLACHER_SUBJECTS:
            for session in ("no_stim", "stim"):
                for suffix in ("vhdr", "eeg", "vmrk"):
                    path = directory / subject / f"{session}.{suffix}"
                    (absent if _absent(path) else present).append(path)
    return present, absent, "explicit"


def _check_kai_miller_nback(root: Path, config: dict):
    directory = _dataset_root(root, "kai_miller_nback", config)
    present, absent = [], []
    if directory is not None:
        for subject in MILLER_SUBJECTS:
            paths = (
                directory / "data" / f"{subject}_nback.mat",
                directory / "locs" / f"{subject}_electrodes.mat",
            )
            for path in paths:
                (absent if _absent(path) else present).append(path)
    return present, absent, "explicit"


def _check_ds005034(root: Path, config: dict):
    directory = _dataset_root(root, "ds005034", config)
    present, absent = [], []
    if directory is not None and directory.is_dir():
        for sidecar in sorted(directory.glob("sub-*/ses-*/eeg/*_eeg.json")):
            stem = sidecar.name[: -len("_eeg.json")]
            for path in (
                sidecar.with_suffix(".set"),
                sidecar.parent / f"{stem}_events.tsv",
                sidecar.parent / f"{stem}_channels.tsv",
            ):
                (absent if _absent(path) else present).append(path)
    return present, absent, "explicit"


def _check_ds006848(root: Path, config: dict):
    directory = _dataset_root(root, "ds006848", config)
    present, absent = [], []
    if directory is not None and directory.is_dir():
        for subject_dir in sorted(directory.glob("sub-*")):
            vhdr_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.vhdr"
            beh_path = subject_dir / "beh" / f"{subject_dir.name}_task-verbalwm_beh.tsv"
            events_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_events.tsv"
            if not (vhdr_path.is_file() and beh_path.is_file() and events_path.is_file()):
                continue
            channels_path = subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_channels.tsv"
            for path in (
                subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.eeg",
                subject_dir / "eeg" / f"{subject_dir.name}_task-verbalwm_eeg.vmrk",
                channels_path,
            ):
                (absent if _absent(path) else present).append(path)
    return present, absent, "explicit"


CORPUS_CHECKS = {
    "dandi_000469": _check_dandi_000469,
    "dandi_001187": _check_dandi_001187,
    "dandi_000673": _check_dandi_000673,
    "dandi_000574": _check_dandi_000574,
    "dandi_000004": _check_dandi_000004,
    "panichello_2024": _check_panichello_2024,
    "watters_2026": _check_watters_2026,
    "inagaki_alm5": _check_inagaki_alm5,
    "macaque_pfc_microstimulation": _check_macaque_pfc_microstimulation,
    "haslacher_clam_tacs": _check_haslacher_clam_tacs,
    "alagapan_phase_stimulation": _check_alagapan_phase_stimulation,
    "ram_ds005489_openloop": _check_ram_bids_stim("ram_ds005489_openloop"),
    "ram_ds005557_closedloop": _check_ram_bids_stim("ram_ds005557_closedloop"),
    "ds004752": _check_ds004752,
    "pfc3": _check_pfc3,
    "pfc4": _check_pfc4,
    "wolff_eeg_impulse": _check_wolff_eeg_impulse,
    "kai_miller_nback": _check_kai_miller_nback,
    "ds005034": _check_ds005034,
    "ds006848": _check_ds006848,
}


def corpus_inventory(root: Path, dataset_key: str, config: dict | None = None) -> dict:
    config = config or load_datasets_config()
    directory = _dataset_root(root, dataset_key, config)
    handler = CORPUS_CHECKS.get(dataset_key)
    if handler is None:
        return {
            "resolved_root": str(directory) if directory else None,
            "mode": "no_handler",
            "required": 0, "present": 0, "absent_paths": [], "absent_total": 0,
            "bytes_on_disk": 0, "status": "unverified",
        }
    root_exists = directory is not None and directory.is_dir()
    present, absent, mode = handler(root, config)
    total_bytes = sum(p.stat().st_size for p in present if p.is_file())
    required = len(present) + len(absent)
    if not root_exists:
        status = "unverified"
    elif len(absent) == 0:
        status = "complete"
    else:
        status = "incomplete"
    absent_str = sorted(str(p) for p in absent)
    return {
        "resolved_root": str(directory) if directory else None,
        "mode": mode,
        "required": required,
        "present": len(present),
        "absent_paths": absent_str[:50],
        "absent_total": len(absent_str),
        "bytes_on_disk": total_bytes,
        "status": status,
    }


def missing_files(root: Path, dataset_keys, config: dict | None = None) -> list[tuple[str, str]]:
    config = config or load_datasets_config()
    found: list[tuple[str, str]] = []
    for dataset_key in dataset_keys:
        handler = CORPUS_CHECKS.get(dataset_key)
        if handler is None:
            raise KeyError(f"no completeness check is registered for corpus {dataset_key!r}")
        _present, absent, _mode = handler(root, config)
        found.extend((dataset_key, str(path)) for path in sorted(absent))
    return found
