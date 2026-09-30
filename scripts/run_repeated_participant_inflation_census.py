#!/usr/bin/env python3
"""Measures, corpus by corpus and artifact by artifact, how much a repeated
patient/session recording structure could inflate an inferential test's
apparent sample size across this repository's result artifacts.

Two independent measurements feed the census:

1. Per human corpus: distinct recording sessions vs. distinct participants,
   counted directly from the staged data (via src/corpus_sessions.py's
   iterators where one exists, or the corpus's own directory layout
   otherwise -- never from a document). ratio = sessions / participants.
   A corpus with ratio 1.0 contributes multiple sessions from the same
   participant to no test in this repository; a corpus with ratio > 1.0
   does, for every test that reports its session count as if it were an
   independent-unit count.

2. Per artifact under results/*.json: every dict node is inspected for a
   key naming a session- or subject-scale count (n_sessions, n_subjects,
   n_patients, n_participants) sitting alongside a key naming inferential
   output (p_value, ci, se, slope, beta, mdd, ...). Where that co-located
   count's corpus is identifiable and that corpus's own ratio is above 1.0,
   the count divided by the ratio is reported as an effective independent n,
   and the square root of the ratio as the standard-error inflation factor
   a test computed at the raw count would be understating.

This is a measurement of exposure, not a verdict. It does not re-run, does
not correct, and does not rank any existing test, corpus, or method; an
artifact appearing here is not thereby shown to be wrong. See the `scope`
field of the written artifact for the complete, standalone statement of
what was and was not measured.

Run:
    python scripts/run_repeated_participant_inflation_census.py
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from project_config import data_root, dataset_path, load_dataset_registry

RESULTS_DIR = ROOT / "results"
OUT_PATH = RESULTS_DIR / "repeated_participant_inflation_census.json"

# ---------------------------------------------------------------------------
# Step 1: per-corpus sessions / participants / ratio, derived from the data.
# ---------------------------------------------------------------------------

# Human corpora only (registry "modality" excludes single_unit macaque/mouse
# releases panichello_2024, watters_2026, inagaki_alm5,
# macaque_pfc_microstimulation, and pfc3 -- none of those record human
# patients, so "repeated participant" does not apply to them).
HUMAN_CORPORA = [
    "dandi_000469", "dandi_001187", "dandi_000673", "dandi_000574",
    "ram_ds005489_openloop", "ram_ds005557_closedloop", "ds004752",
    "wolff_eeg_impulse", "kai_miller_nback", "haslacher_clam_tacs",
    "alagapan_phase_stimulation",
]

# Text tokens that identify a corpus inside an artifact's own JSON content or
# filename. Matched case-insensitively as substrings. Kept corpus-specific
# (e.g. not the generic construct "n-back") so a match is not a guess.
CORPUS_TOKENS = {
    "dandi_000469": ["000469", "dandi_000469"],
    "dandi_001187": ["001187", "dandi_001187"],
    "dandi_000673": ["000673", "dandi_000673"],
    "dandi_000574": ["000574", "dandi_000574", "boran"],
    "ram_ds005489_openloop": ["ds005489", "ram_ds005489_openloop"],
    "ram_ds005557_closedloop": ["ds005557", "ram_ds005557_closedloop"],
    "ds004752": ["ds004752"],
    "wolff_eeg_impulse": ["wolff_eeg_impulse", "dynamic_hidden_states", "wolff"],
    "kai_miller_nback": ["kai_miller_nback", "kai_miller", "miller_nback"],
    "haslacher_clam_tacs": ["haslacher_clam_tacs", "haslacher", "clam_tacs", "clam-tacs"],
    "alagapan_phase_stimulation": ["alagapan_phase_stimulation", "alagapan"],
}


def _bids_session_counts(local_path: str) -> dict:
    root = dataset_path_for(local_path)
    if root is None or not root.is_dir():
        return {"enumerable": False, "reason": f"directory not found at {root}"}
    subjects = sorted(p.name for p in root.glob("sub-*") if p.is_dir())
    sessions = sorted(str(p.relative_to(root)) for p in root.glob("sub-*/ses-*") if p.is_dir())
    if not subjects:
        return {"enumerable": False, "reason": f"no sub-* directories under {root}"}
    n_sessions = len(sessions) if sessions else len(subjects)
    method = "BIDS sub-*/ses-* directory count" if sessions else "BIDS sub-* directory count (no ses-* subdirectories found)"
    return {"enumerable": True, "n_sessions": n_sessions, "n_participants": len(subjects), "method": method}


def dataset_path_for(local_path: str) -> Path | None:
    root = data_root(required=False)
    return None if root is None else root / local_path


def _corpus_000469() -> dict:
    from corpus_sessions import iter_dandi_000469  # noqa
    root = data_root(required=False)
    if root is None or not (root / "000469").is_dir():
        return {"enumerable": False, "reason": "000469 directory not found under the data root"}
    pairs = {(row["patient"], row["session"]) for row in iter_dandi_000469(root)}
    participants = {p for p, _ in pairs}
    if not pairs:
        return {"enumerable": False, "reason": "iter_dandi_000469 yielded no sessions (all excluded by its own QC)"}
    return {"enumerable": True, "n_sessions": len(pairs), "n_participants": len(participants),
            "method": "src/corpus_sessions.py iter_dandi_000469, deduplicated over its per-region yields"}


def _corpus_000574() -> dict:
    from corpus_sessions import iter_dandi_000574  # noqa
    root = data_root(required=False)
    if root is None or not (root / "000574").is_dir():
        return {"enumerable": False, "reason": "000574 directory not found under the data root"}
    pairs = {(row["patient"], row["session"]) for row in iter_dandi_000574(root)}
    participants = {p for p, _ in pairs}
    if not pairs:
        return {"enumerable": False, "reason": "iter_dandi_000574 yielded no sessions (all excluded by its own QC)"}
    return {"enumerable": True, "n_sessions": len(pairs), "n_participants": len(participants),
            "method": "src/corpus_sessions.py iter_dandi_000574, deduplicated over its per-region yields"}


def _corpus_001187_000673(release: str) -> dict:
    provenance_dir = ROOT / "provenance"
    registry_path = provenance_dir / "canonical_recording_registry.json"
    primary_path = provenance_dir / "canonical_primary_records.json"
    if not (registry_path.exists() and primary_path.exists()):
        return {"enumerable": False, "reason": f"{registry_path} / {primary_path} not found"}
    from run_human_drift_spine_001187_000673 import canonical_sessions  # noqa
    rows = [r for r in canonical_sessions(provenance_dir) if r["primary_release"] == release]
    if not rows:
        return {"enumerable": False, "reason": f"no canonical rows with primary_release=={release!r}"}
    participants = {r["patient"] for r in rows}
    return {"enumerable": True, "n_sessions": len(rows), "n_participants": len(participants),
            "method": ("provenance/canonical_{primary_records,recording_registry}.json "
                       f"canonical_sessions() rows with primary_release=={release!r} -- the "
                       "project's own deduplicated view of this release, not its raw session count, "
                       "since the release overlaps dandi_" + ("000673" if release == "001187" else "001187"))}


def _corpus_wolff() -> dict:
    root = dataset_path_for(load_dataset_registry()["datasets"]["wolff_eeg_impulse"]["local_path"])
    if root is None or not root.is_dir():
        return {"enumerable": False, "reason": f"directory not found at {root}"}
    exp1 = sorted(root.glob("Dynamic_hidden_states_exp1_*.mat"))
    exp2 = sorted(root.glob("Dynamic_hidden_states_exp2_*.mat"))
    n = len(exp1) + len(exp2)
    if n == 0:
        return {"enumerable": False, "reason": f"no Dynamic_hidden_states_exp{{1,2}}_*.mat files under {root}"}
    return {"enumerable": True, "n_sessions": n, "n_participants": n,
            "method": (f"file count: {len(exp1)} exp1 + {len(exp2)} exp2 .mat files, each one participant's "
                       "one session, matching scripts/run_wolff_impulse_pipeline.py's own per-file subject "
                       "pooling; the mismatched exp1/exp2 counts rule out a shared participant numbering "
                       "across experiments, so the two pools are counted as independent participants here")}


def _corpus_kai_miller() -> dict:
    local = load_dataset_registry()["datasets"]["kai_miller_nback"]["local_path"]
    root = dataset_path_for(local)
    data_dir = None if root is None else root / "data"
    if data_dir is None or not data_dir.is_dir():
        return {"enumerable": False, "reason": f"directory not found at {data_dir}"}
    files = sorted(data_dir.glob("*_nback.mat"))
    if not files:
        return {"enumerable": False, "reason": f"no *_nback.mat files under {data_dir}"}
    return {"enumerable": True, "n_sessions": len(files), "n_participants": len(files),
            "method": "file count: one *_nback.mat file per patient (scripts/run_miller_drift_spine.py "
                      "confirms exactly one session per of its 4 patients)"}


def _corpus_haslacher() -> dict:
    root = dataset_path("haslacher_clam_tacs", required=False)
    if root is None or not root.is_dir():
        return {"enumerable": False, "reason": f"directory not found at {root}"}
    participant_dirs = sorted(p for p in root.glob("PA*") if (p / "no_stim.vhdr").exists())
    if not participant_dirs:
        return {"enumerable": False, "reason": f"no PA*/no_stim.vhdr participant directories under {root}"}
    return {"enumerable": True, "n_sessions": len(participant_dirs), "n_participants": len(participant_dirs),
            "method": "directory count: one PA* participant directory per session (no_stim/stim are "
                      "stimulation conditions within that one session, per "
                      "scripts/run_haslacher_phase_omega.py's own per-participant loop)"}


def _corpus_alagapan() -> dict:
    root = dataset_path("alagapan_phase_stimulation", required=False)
    baseline_dir = None if root is None else root / "iEEG Data" / "Baseline"
    if baseline_dir is None or not baseline_dir.is_dir():
        return {"enumerable": False, "reason": f"directory not found at {baseline_dir}"}
    patients = sorted({p.name.split("_")[0] for p in baseline_dir.glob("*_SMS_Baseline_Epoched.set")})
    if not patients:
        return {"enumerable": False, "reason": f"no *_SMS_Baseline_Epoched.set files under {baseline_dir}"}
    return {"enumerable": True, "n_sessions": len(patients), "n_participants": len(patients),
            "method": "filename-prefix count: one patient prefix per session (Baseline/Stimulation are "
                      "conditions within that one session, per scripts/run_alagapan_phase_omega.py's "
                      "PATIENTS list)"}


def build_corpus_registry() -> dict:
    registry = load_dataset_registry()["datasets"]
    out = {}
    for key in HUMAN_CORPORA:
        if key not in registry:
            out[key] = {"enumerable": False, "reason": "not present in config/datasets.json"}
            continue
        try:
            if key == "dandi_000469":
                result = _corpus_000469()
            elif key == "dandi_000574":
                result = _corpus_000574()
            elif key in ("dandi_001187", "dandi_000673"):
                result = _corpus_001187_000673("001187" if key == "dandi_001187" else "000673")
            elif key in ("ram_ds005489_openloop", "ram_ds005557_closedloop", "ds004752"):
                result = _bids_session_counts(registry[key]["local_path"])
            elif key == "wolff_eeg_impulse":
                result = _corpus_wolff()
            elif key == "kai_miller_nback":
                result = _corpus_kai_miller()
            elif key == "haslacher_clam_tacs":
                result = _corpus_haslacher()
            elif key == "alagapan_phase_stimulation":
                result = _corpus_alagapan()
            else:
                result = {"enumerable": False, "reason": "no enumeration method implemented for this corpus key"}
        except Exception as exc:  # a data-layout surprise is a not-enumerable finding, not a crash
            result = {"enumerable": False, "reason": f"{type(exc).__name__}: {exc}"}
        if result.get("enumerable") and result["n_participants"] > 0:
            result["ratio_sessions_per_participant"] = result["n_sessions"] / result["n_participants"]
        out[key] = result
    return out


# ---------------------------------------------------------------------------
# Step 2/3: scan results/*.json for session-scale counts co-located with
# inferential output, and compute the effective-n measurement where the
# corpus is identifiable and its ratio exceeds 1.0.
# ---------------------------------------------------------------------------

# Key classification is token-based (split on "_"), not raw substring
# containment: a raw substring test matches "n_subject" inside the unrelated
# word run "...betwee[n_subject]ect", and "se"/"ci" as raw substrings match
# inside ordinary words like "response" or "specificity". Splitting on "_"
# and matching whole tokens avoids both classes of false positive while
# still catching this repo's actual naming convention (n_sessions,
# n_multisite_subjects, cluster_ci, trial_level_p_for_reference, ...).
COUNT_NOUNS = {"session", "sessions", "subject", "subjects", "patient", "patients",
               "participant", "participants"}
# "series" and "cell(s)" are handled separately -- the brief flags "series"
# as the paradigm case of an ambiguous count, and "cell" in this
# repository's artifacts denotes recorded units/electrodes, not sessions.
AMBIGUOUS_COUNT_NOUNS = {"series", "cell", "cells"}
# A count noun immediately preceded by one of these names a STUDY DESIGN
# ("between-subject", "within-session"), not a count of that unit -- the
# one false positive this scan actually hit (n_null_draws_..._between_subject).
DESIGN_MODIFIER_PRECEDERS = {"between", "within", "across"}

LONG_INFERENTIAL_SUBSTRINGS = ("p_value", "confidence_interval", "standard_error", "minimum_detectable")
SHORT_INFERENTIAL_TOKENS = {"p", "ci", "se", "beta", "mdd", "slope"}


def _has_count_noun(key: str, nouns: set[str]) -> bool:
    tokens = key.lower().split("_")
    if "n" not in tokens:
        return False
    for i, tok in enumerate(tokens):
        if tok in nouns and (i == 0 or tokens[i - 1] not in DESIGN_MODIFIER_PRECEDERS):
            return True
    return False


def _is_count_key(key: str) -> bool:
    return _has_count_noun(key, COUNT_NOUNS)


def _is_ambiguous_count_key(key: str) -> bool:
    return _has_count_noun(key, AMBIGUOUS_COUNT_NOUNS)


def _is_inferential_key(key: str) -> bool:
    lk = key.lower()
    if any(s in lk for s in LONG_INFERENTIAL_SUBSTRINGS):
        return True
    return any(t in SHORT_INFERENTIAL_TOKENS for t in lk.split("_"))


DATASET_HINT_KEYS = {"dataset", "corpus", "release", "dataset_key", "corpus_key"}


def _corpus_from_text(text: str) -> list[str]:
    hits = []
    for corpus, tokens in CORPUS_TOKENS.items():
        if any(tok in text for tok in tokens):
            hits.append(corpus)
    return hits


def _corpus_from_value(value: str) -> list[str]:
    lv = value.lower()
    return [c for c, tokens in CORPUS_TOKENS.items() if any(tok in lv for tok in tokens)]


def _walk(node, path: str, colocations: list, count_hits: list, ambiguous_hits: list, strings: list):
    """Recursively visits every dict node once; records, per dict node
    encountered, its own count keys / inferential keys / ambiguous keys, and
    any co-location of a count key with an inferential key in that same node.

    ``strings`` collects every dict/list-key and every string-typed leaf
    value seen (never numeric values) -- corpus identification searches only
    this pool, not the raw serialized JSON text, because a numeric id like
    "000574" can appear by coincidence inside an unrelated float's decimal
    digits (observed: -0.000574422011091702 in a different corpus's
    artifact), which a naive whole-text substring search would misread as
    that corpus being named."""
    if isinstance(node, dict):
        keys = list(node.keys())
        strings.extend(keys)
        node_count_keys = [k for k in keys if _is_count_key(k)]
        node_ambig_keys = [k for k in keys if _is_ambiguous_count_key(k)]
        node_inf_keys = [k for k in keys if _is_inferential_key(k)]
        for k in node_count_keys:
            count_hits.append({"path": f"{path}.{k}" if path else k, "key": k, "value": node[k]})
        for k in node_ambig_keys:
            ambiguous_hits.append({"path": f"{path}.{k}" if path else k, "key": k, "value": node[k]})
        if node_count_keys and node_inf_keys:
            hint_corpora = []
            for hk in DATASET_HINT_KEYS:
                if hk in node and isinstance(node[hk], str):
                    hint_corpora.extend(_corpus_from_value(node[hk]))
            colocations.append({
                "path": path or "$",
                "count_keys": [{"key": k, "value": node[k]} for k in node_count_keys],
                "inferential_keys": node_inf_keys,
                "node_level_corpus_hint": sorted(set(hint_corpora)),
            })
        for k, v in node.items():
            _walk(v, f"{path}.{k}" if path else k, colocations, count_hits, ambiguous_hits, strings)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _walk(v, f"{path}[{i}]", colocations, count_hits, ambiguous_hits, strings)
    elif isinstance(node, str):
        strings.append(node)


def scan_artifact(path: Path, corpus_registry: dict) -> dict:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {"file": path.name, "status": "unreadable", "reason": f"{type(exc).__name__}: {exc}"}

    colocations, count_hits, ambiguous_hits, strings = [], [], [], []
    _walk(data, "", colocations, count_hits, ambiguous_hits, strings)
    strings.append(path.name)

    file_text_lower = " ".join(strings).lower()
    file_level_corpora = _corpus_from_text(file_text_lower)

    measurements = []
    for coloc in colocations:
        # prefer a node-level dataset/corpus field; fall back to the whole
        # file naming exactly one human corpus unambiguously.
        if len(coloc["node_level_corpus_hint"]) == 1:
            corpus = coloc["node_level_corpus_hint"][0]
            corpus_source = "node_level_field"
        elif len(coloc["node_level_corpus_hint"]) == 0 and len(file_level_corpora) == 1:
            corpus = file_level_corpora[0]
            corpus_source = "file_level_unique_mention"
        else:
            coloc["corpus_identifiable"] = False
            coloc["corpus_ambiguity_reason"] = (
                f"node-level hints={coloc['node_level_corpus_hint']!r}, "
                f"file-level mentions={file_level_corpora!r}"
            )
            continue
        coloc["corpus_identifiable"] = True
        coloc["corpus"] = corpus
        coloc["corpus_identification_method"] = corpus_source

        corpus_info = corpus_registry.get(corpus, {})
        if not corpus_info.get("enumerable"):
            coloc["corpus_inflation_factor_available"] = False
            continue
        ratio = corpus_info["ratio_sessions_per_participant"]
        coloc["corpus_inflation_factor_available"] = True
        coloc["corpus_inflation_factor"] = ratio
        if ratio <= 1.0:
            continue
        se_inflation = math.sqrt(ratio)
        for ck in coloc["count_keys"]:
            value = ck["value"]
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                continue
            effective_n = value / ratio
            measurements.append({
                "file": path.name,
                "node_path": coloc["path"],
                "count_key": ck["key"],
                "reported_count": value,
                "corpus": corpus,
                "corpus_identification_method": corpus_source,
                "inflation_factor": ratio,
                "effective_n": effective_n,
                "se_inflation_factor": se_inflation,
                "inferential_keys_in_same_node": coloc["inferential_keys"],
                "reduction_fraction": 1.0 - effective_n / value,
            })

    return {
        "file": path.name,
        "status": "scanned",
        "n_count_key_hits": len(count_hits),
        "n_ambiguous_count_key_hits": len(ambiguous_hits),
        "n_colocations": len(colocations),
        "file_level_corpus_mentions": sorted(set(file_level_corpora)),
        "colocations": colocations,
        "measurements": measurements,
    }


def summarize(per_file: list[dict], corpus_registry: dict) -> dict:
    n_scanned = len(per_file)
    n_with_count = sum(1 for f in per_file if f.get("n_count_key_hits", 0) > 0)
    n_with_colocation = sum(1 for f in per_file if f.get("n_colocations", 0) > 0)
    repeat_heavy_corpora = {c for c, r in corpus_registry.items()
                             if r.get("enumerable") and r["ratio_sessions_per_participant"] > 1.0}
    files_on_repeat_heavy = [f for f in per_file if f.get("measurements")]
    all_measurements = [m for f in per_file for m in f.get("measurements", [])]
    reductions = sorted(m["reduction_fraction"] for m in all_measurements)

    def _pct(sorted_vals, q):
        if not sorted_vals:
            return None
        idx = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
        return sorted_vals[idx]

    largest_exposure = sorted(
        all_measurements, key=lambda m: (m["reported_count"] - m["effective_n"]), reverse=True
    )[:10]
    largest_exposure_files = []
    for f in files_on_repeat_heavy:
        gap = sum(m["reported_count"] - m["effective_n"] for m in f["measurements"])
        largest_exposure_files.append({"file": f["file"], "total_effective_n_gap": gap,
                                        "n_measurements": len(f["measurements"])})
    largest_exposure_files.sort(key=lambda x: x["total_effective_n_gap"], reverse=True)

    return {
        "n_artifacts_scanned": n_scanned,
        "n_artifacts_with_session_scale_count": n_with_count,
        "n_artifacts_with_count_inferential_colocation": n_with_colocation,
        "n_artifacts_with_measurement_on_repeat_heavy_corpus": len(files_on_repeat_heavy),
        "repeat_heavy_corpora": sorted(repeat_heavy_corpora),
        "n_measurements": len(all_measurements),
        "reduction_fraction_distribution": {
            "min": reductions[0] if reductions else None,
            "median": _pct(reductions, 0.5),
            "p90": _pct(reductions, 0.9),
            "max": reductions[-1] if reductions else None,
        },
        "largest_exposure_measurements": [
            {"file": m["file"], "node_path": m["node_path"], "count_key": m["count_key"],
             "reported_count": m["reported_count"], "effective_n": m["effective_n"],
             "corpus": m["corpus"], "inflation_factor": m["inflation_factor"]}
            for m in largest_exposure
        ],
        "largest_exposure_artifacts": largest_exposure_files[:10],
    }


SCOPE = (
    "This artifact measures two things and asserts nothing beyond them. First, for each human "
    "corpus this project stages, the ratio of distinct recording sessions to distinct participants, "
    "counted directly from the staged data (or recorded as not-enumerable with a stated reason when "
    "the data could not be counted); the unit of that ratio is sessions per participant, so a ratio "
    "of 1.0 means every session in that corpus comes from a different participant, and a ratio above "
    "1.0 means some participants contributed more than one session. Second, for every results/*.json "
    "artifact in this repository, every place where a key naming a session- or subject-scale count "
    "sits in the same JSON object as a key naming inferential output (a p-value, a confidence interval, "
    "a standard error, a slope, a minimum detectable effect, and similar), together with, where the "
    "corpus of that count is identifiable and that corpus's own ratio exceeds 1.0, the count divided by "
    "the ratio (an effective independent n) and the square root of the ratio (a standard-error inflation "
    "factor). This is a census of EXPOSURE: it identifies where a session-scale count and an inferential "
    "statistic happen to sit together and by how much the count could shrink if only distinct "
    "participants were counted. It does not re-run, re-fit, or correct any test in any artifact it "
    "scans, does not decide whether a given test in fact treated sessions as independent, and does not "
    "compute a corrected p-value, interval, or effect size for anything it finds. An artifact appearing "
    "in this census -- including in its largest-exposure lists -- is not thereby shown to be wrong: it "
    "may already account for repeated participants by another method (a cluster-robust or "
    "patient-level estimator) that this scan cannot see, or the co-located count and inferential value "
    "may not in fact both feed the same statistical claim. Ambiguous count keys ('series', 'cell') are "
    "recorded separately from the main tally rather than assumed to mean sessions or subjects."
)


def main() -> None:
    corpus_registry = build_corpus_registry()
    artifact_paths = sorted(RESULTS_DIR.glob("*.json"))
    per_file = [scan_artifact(p, corpus_registry) for p in artifact_paths
                if p.name != "repeated_participant_inflation_census.json"]
    summary = summarize(per_file, corpus_registry)

    out = {
        "scope": SCOPE,
        "corpus_registry": corpus_registry,
        "summary": summary,
        "per_artifact": [
            {k: v for k, v in f.items()} for f in per_file
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
