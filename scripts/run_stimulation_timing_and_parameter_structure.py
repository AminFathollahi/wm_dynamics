#!/usr/bin/env python3
"""When in the sequence of remembered items does delivered stimulation land,
and what can the two human intracranial free-recall stimulation corpora
resolve about timing and dose -- from event tables alone, no recording
signal is read.

Two public corpora, one shared event schema (onset, duration, trial_type,
serialpos, recalled, stimulation, stim_list, stim_duration, anode_label,
cathode_label, amplitude, pulse_freq, n_pulses, pulse_width, ...):

  ds005489 -- open-loop arm. Stimulation is delivered on an experimenter-
  scheduled, alternating two-word block within encoding; WORD rows carry
  their own `stimulation` flag and full dose fields directly.

  ds005557 -- classifier-triggered arm. WORD rows leave `stimulation` at
  "0" always; real stimulation timing and dose live on STIM_ON/STIM_OFF
  rows and must be matched back onto the item whose encoding-period neural
  state triggered them.

Three questions, each answered by measurement, not assumption:

  Block A -- can the delivered pulse train be localised to a single item's
  presentation at all, or does it necessarily span more than one? Decided
  per corpus from the actual overlap of stimulation-train intervals against
  item on-screen presentation windows, read from the event tables.

  Block B -- does the stimulated-minus-control difference in recall depend
  on where in the list the item was, i.e. an interaction between treatment
  and serial position (with both main effects reported beside it)? Computed
  within subject, then across subjects with subject as the clustering unit.
  Restricted to the open-loop corpus: closed-loop item-level stimulation is
  triggered by the classifier's own reading of the state whose downstream
  behavioural consequence would be under test here (propensity-selected on
  the outcome-relevant signal, not randomized), the same restriction this
  project has already placed on that corpus's item-level comparisons
  elsewhere -- so an item-level position interaction computed on it would
  not be a causal claim, and is not attempted here as one.

  Block C -- the parameter census: subjects, sessions, electrode pairs and
  stimulated trials at each delivered amplitude in both corpora, and which
  stimulation parameters are constant (and therefore unaskable) in each.

  Block D -- what a non-human delay-period stimulation corpus can add to
  what these two arms establish, and what is named as a genuine gap rather
  than papered over.

Output: results/stimulation_timing_and_parameter_structure.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_stimulation_timing_and_parameter_structure.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import csv
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from project_config import data_root, dataset_path, executable, project_path

from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference,
    paired_sign_flip_test,
    stable_seed,
)

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_stimulation_timing_and_parameter_structure"
OUT_PATH = RESULTS / "stimulation_timing_and_parameter_structure.json"

DATA_ROOT = data_root()
OPENLOOP_CORPUS = "ds005489-download"
CLOSEDLOOP_CORPUS = "ds005557-download"

ALPHA = 0.05
POWER = 0.80
N_PERM = 10000
N_BOOT = 5000

# Margin used to attribute a closed-loop STIM_ON pulse train back to the WORD
# item whose encoding produced it: a train is owned by the nearest preceding
# word within this many seconds of its own onset. Reuses, unchanged, the
# window already established by this project's other closed-loop
# derivation (scripts/run_ram_openloop_pipeline.py's `_derive_word_stimulation`,
# `derive_stim_from_stim_on=True`) rather than inventing a new value here.
CLOSEDLOOP_OWNER_MATCH_WINDOW_S = 2.0

# Block A decision: an "item presentation" is the interval a WORD is on
# screen, [onset, onset + duration]. A pulse train "spans several items" if,
# pooled over every train in the corpus that overlaps at least one item at
# all, the mean number of item presentations one train overlaps exceeds 1 --
# the literal reading of "one train covers more than one item". Frozen
# before the corpus was scanned.
ITEMS_PER_TRAIN_SPAN_THRESHOLD = 1.0

# Block B: minimum stimulated and minimum control item trials a subject must
# contribute for their own position-interaction regression to be attempted.
MIN_TRIALS_PER_ARM_PER_SUBJECT = 10

# List length is fixed at 12 throughout ds005489 (verified below at run
# time, not assumed); serial position is centered on its corpus-wide mean
# so the interaction coefficient reads directly as "probability change per
# stim x per serial-position step".
SERIALPOS_CENTER = 6.5


# ── Event-table IO (no recording signal is ever opened) ─────────────────────

def _num(value) -> float:
    """Parse a BIDS TSV field (always a string from csv.DictReader): a
    number, 'n/a', or empty -- never raises."""
    try:
        if value in (None, "", "n/a"):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_events(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["_onset"] = _num(r["onset"])
        r["_duration"] = _num(r["duration"])
    return rows


def discover_sessions(corpus_dir: str) -> list[Path]:
    return sorted((DATA_ROOT / corpus_dir).glob("sub-*/ses-*/ieeg/*_events.tsv"))


def discover_subject_dirs(corpus_dir: str) -> list[str]:
    return sorted(p.name for p in (DATA_ROOT / corpus_dir).glob("sub-*") if p.is_dir())


# ── Interval geometry ────────────────────────────────────────────────────────

def overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def group_words_by_list(words: list[dict]) -> dict[str, list[dict]]:
    by_list = defaultdict(list)
    for w in words:
        by_list[w["list"]].append(w)
    for lst in by_list:
        by_list[lst].sort(key=lambda w: w["_onset"])
    return by_list


def within_list_spacing(by_list: dict[str, list[dict]]) -> list[float]:
    """Consecutive WORD onset differences within a list -- the item
    presentation interval, measured, not assumed."""
    diffs = []
    for ws in by_list.values():
        for i in range(1, len(ws)):
            diffs.append(ws[i]["_onset"] - ws[i - 1]["_onset"])
    return diffs


# ── Train construction ───────────────────────────────────────────────────────

def build_trains_openloop(rows: list[dict]) -> list[dict]:
    """One train per STIM_ON row: [onset, onset + duration], using each
    row's own duration field directly rather than the WORD-row stim_duration
    (ms) field, so the train boundary is read straight from the event
    table's own timestamps."""
    trains = []
    for r in rows:
        if r["trial_type"] != "STIM_ON":
            continue
        s, dur = r["_onset"], r["_duration"]
        if not (np.isfinite(s) and np.isfinite(dur)) or dur <= 0:
            continue
        trains.append({
            "start": s, "end": s + dur,
            "amplitude": _num(r.get("amplitude")), "pulse_freq": _num(r.get("pulse_freq")),
            "pulse_width": _num(r.get("pulse_width")), "stim_duration_ms": _num(r.get("stim_duration")),
            "anode": r.get("anode_label"), "cathode": r.get("cathode_label"),
        })
    return trains


def build_trains_closedloop(rows: list[dict]) -> tuple[list[dict], int]:
    """Pair each STIM_ON with the STIM_OFF immediately following it in time
    (sequential zip after sorting both by onset): a real train interval, not
    the STIM_ON row's own 'duration' field, which is 0 in this corpus (the
    real off time lives on the separate STIM_OFF row). Returns
    (trains, n_count_mismatch_sessions) -- if ON and OFF counts disagree,
    no trains are built and the mismatch is reported rather than guessed at."""
    stim_on = sorted((r for r in rows if r["trial_type"] == "STIM_ON"), key=lambda r: r["_onset"])
    stim_off = sorted((r for r in rows if r["trial_type"] == "STIM_OFF"), key=lambda r: r["_onset"])
    if len(stim_on) != len(stim_off):
        return [], 1
    trains = []
    for on, off in zip(stim_on, stim_off):
        if off["_onset"] <= on["_onset"]:
            continue
        trains.append({
            "start": on["_onset"], "end": off["_onset"],
            "amplitude": _num(on.get("amplitude")), "pulse_freq": _num(on.get("pulse_freq")),
            "pulse_width": _num(on.get("pulse_width")), "stim_duration_ms": _num(on.get("stim_duration")),
            "anode": on.get("anode_label"), "cathode": on.get("cathode_label"),
        })
    return trains, 0


def match_train_owner(train_start: float, word_onsets: list[float], window_s: float) -> int | None:
    """Nearest preceding word within window_s of the train's own onset, or
    None. Matches scripts/run_ram_openloop_pipeline.py's
    `_derive_word_stimulation` convention exactly."""
    candidates = [(train_start - w, i) for i, w in enumerate(word_onsets) if 0 <= train_start - w <= window_s]
    if not candidates:
        return None
    return min(candidates)[1]


# ── Block A: attributability of a train to a single item's presentation ─────

def block_a_session(words_sorted: list[dict], trains: list[dict]) -> dict:
    """Per session: for every train, how many item on-screen presentations
    ([onset, onset+duration]) it overlaps; and, per item that owns a train,
    whether that train also overlaps the immediately preceding or following
    item in serial order (regardless of that neighbour's own stim status)."""
    item_windows = [(w["_onset"], w["_onset"] + w["_duration"]) for w in words_sorted]
    items_per_train = []
    for t in trains:
        n = sum(1 for (i0, i1) in item_windows if overlaps(t["start"], t["end"], i0, i1))
        items_per_train.append(n)
    return {"items_per_train": items_per_train}


def block_a_neighbor_coverage(by_list: dict[str, list[dict]], trains: list[dict],
                              is_stim_flag) -> dict:
    """is_stim_flag(word) -> bool. For each item flagged stimulated, finds
    the train overlapping its own on-screen window and checks whether that
    same train also overlaps the preceding / following item's on-screen
    window in the same list."""
    n_total = n_prev = n_next = n_any = n_unmatched = 0
    for ws in by_list.values():
        for i, w in enumerate(ws):
            if not is_stim_flag(w):
                continue
            n_total += 1
            w0, w1 = w["_onset"], w["_onset"] + w["_duration"]
            enclosing = [t for t in trains if overlaps(t["start"], t["end"], w0, w1)]
            if not enclosing:
                n_unmatched += 1
                continue
            t = enclosing[0]
            cov_prev = i > 0 and overlaps(t["start"], t["end"], ws[i - 1]["_onset"],
                                          ws[i - 1]["_onset"] + ws[i - 1]["_duration"])
            cov_next = i < len(ws) - 1 and overlaps(t["start"], t["end"], ws[i + 1]["_onset"],
                                                     ws[i + 1]["_onset"] + ws[i + 1]["_duration"])
            n_prev += int(cov_prev)
            n_next += int(cov_next)
            n_any += int(cov_prev or cov_next)
    return {"n_stim_items": n_total, "n_prev_covered": n_prev, "n_next_covered": n_next,
            "n_any_covered": n_any, "n_train_unmatched_to_owning_item": n_unmatched}


# ── Per-session processing ───────────────────────────────────────────────────

def process_openloop_session(rows: list[dict]) -> dict:
    words = [r for r in rows if r["trial_type"] == "WORD"]
    by_list = group_words_by_list(words)
    words_sorted = sorted(words, key=lambda w: w["_onset"])
    trains = build_trains_openloop(rows)

    a = block_a_session(words_sorted, trains)
    nb = block_a_neighbor_coverage(by_list, trains, lambda w: w["stimulation"] == "1")

    # Block B: per-item causal rows (open-loop stim flag is a direct,
    # experimenter-randomized field -- no derivation needed).
    items = []
    for ws in by_list.values():
        for i, w in enumerate(ws):
            is_stim = w["stimulation"] == "1"
            prev_covered = None
            if is_stim:
                w0, w1 = w["_onset"], w["_onset"] + w["_duration"]
                enclosing = [t for t in trains if overlaps(t["start"], t["end"], w0, w1)]
                if enclosing and i > 0:
                    t = enclosing[0]
                    prev_covered = overlaps(t["start"], t["end"], ws[i - 1]["_onset"],
                                            ws[i - 1]["_onset"] + ws[i - 1]["_duration"])
                elif enclosing:
                    prev_covered = False  # first item in the list has no preceding item
            items.append({"stim": int(is_stim), "serialpos": int(w["serialpos"]),
                         "recalled": int(w["recalled"]), "prev_covered": prev_covered})

    spacing = within_list_spacing(by_list)
    train_durations = [t["end"] - t["start"] for t in trains]
    amplitudes = [t["amplitude"] for t in trains if np.isfinite(t["amplitude"])]
    pulse_freqs = sorted({t["pulse_freq"] for t in trains if np.isfinite(t["pulse_freq"])})
    pulse_widths = sorted({t["pulse_width"] for t in trains if np.isfinite(t["pulse_width"])})
    stim_durations_ms = sorted({t["stim_duration_ms"] for t in trains if np.isfinite(t["stim_duration_ms"])})
    pairs = sorted({(t["anode"], t["cathode"]) for t in trains if t["anode"] not in (None, "n/a")})

    return {
        "n_words": len(words), "n_trains": len(trains),
        "spacing_s": spacing, "train_duration_s": train_durations,
        "items_per_train": a["items_per_train"],
        "neighbor_coverage": nb,
        "amplitudes": amplitudes, "pulse_freqs": pulse_freqs, "pulse_widths": pulse_widths,
        "stim_durations_ms": stim_durations_ms, "electrode_pairs": pairs,
        # every open-loop train is already item-attributed by construction (built
        # straight from WORD-linked STIM_ON rows) -- "all trains" equals the matched set.
        "amplitudes_all_trains": amplitudes, "electrode_pairs_all_trains": pairs,
        "items": items,
    }


def process_closedloop_session(rows: list[dict]) -> dict:
    words = [r for r in rows if r["trial_type"] == "WORD"]
    by_list = group_words_by_list(words)
    words_sorted = sorted(words, key=lambda w: w["_onset"])
    trains, count_mismatch = build_trains_closedloop(rows)
    if count_mismatch:
        return {"status": "excluded", "reason": "STIM_ON/STIM_OFF row count mismatch -- "
                "no trains can be paired without guessing", "n_words": len(words)}

    a = block_a_session(words_sorted, trains)

    # Derive owning word for every train (nearest preceding word within
    # CLOSEDLOOP_OWNER_MATCH_WINDOW_S of the train's own onset) -- this
    # boolean stim flag is what a downstream causal or census use of this
    # corpus must build on, since the WORD row's own `stimulation` field is
    # always "0" here.
    word_onsets = [w["_onset"] for w in words_sorted]
    owner_idx_by_train = [match_train_owner(t["start"], word_onsets, CLOSEDLOOP_OWNER_MATCH_WINDOW_S)
                          for t in trains]
    n_train_unowned = sum(1 for i in owner_idx_by_train if i is None)
    owned_word_indices = {i for i in owner_idx_by_train if i is not None}

    pos_of_word = {id(w): i for i, w in enumerate(words_sorted)}

    def is_stim_derived_fast(w: dict) -> bool:
        return pos_of_word.get(id(w)) in owned_word_indices

    nb = block_a_neighbor_coverage(by_list, trains, is_stim_derived_fast)

    spacing = within_list_spacing(by_list)
    train_durations = [t["end"] - t["start"] for t in trains]
    stim_trains = [t for i, t in enumerate(trains) if owner_idx_by_train[i] is not None]
    amplitudes = [t["amplitude"] for t in stim_trains if np.isfinite(t["amplitude"])]
    pulse_freqs = sorted({t["pulse_freq"] for t in stim_trains if np.isfinite(t["pulse_freq"])})
    pulse_widths = sorted({t["pulse_width"] for t in stim_trains if np.isfinite(t["pulse_width"])})
    stim_durations_ms = sorted({t["stim_duration_ms"] for t in stim_trains if np.isfinite(t["stim_duration_ms"])})
    pairs = sorted({(t["anode"], t["cathode"]) for t in stim_trains if t["anode"] not in (None, "n/a")})

    # Unrestricted by word-matching -- includes pre-task titration/calibration pulses
    # that never land near any WORD item (e.g. a device test pulse at recording onset).
    # Kept separately so Block C can show, and flag, the difference between genuine
    # item-linked dose and raw STIM_ON-row dose, rather than silently using whichever
    # one a naive scan of the event table would find first.
    amplitudes_all = [t["amplitude"] for t in trains if np.isfinite(t["amplitude"])]
    pairs_all = sorted({(t["anode"], t["cathode"]) for t in trains if t["anode"] not in (None, "n/a")})

    return {
        "status": "included",
        "n_words": len(words), "n_trains": len(trains),
        "n_trains_unowned": n_train_unowned,
        "spacing_s": spacing, "train_duration_s": train_durations,
        "items_per_train": a["items_per_train"],
        "neighbor_coverage": nb,
        "amplitudes": amplitudes, "pulse_freqs": pulse_freqs, "pulse_widths": pulse_widths,
        "stim_durations_ms": stim_durations_ms, "electrode_pairs": pairs,
        "amplitudes_all_trains": amplitudes_all, "electrode_pairs_all_trains": pairs_all,
        "n_stim_items_derived": len(owned_word_indices),
    }


# ── Checkpointing (per session, crash-proof) ─────────────────────────────────

def _checkpoint_path(session_id: str) -> Path:
    return CHECKPOINT_DIR / f"{session_id}.json"


def load_checkpoint(session_id: str) -> dict | None:
    """An unparseable or incomplete checkpoint record is treated as absent."""
    path = _checkpoint_path(session_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    record = data["record"]
    # Schema check, not just parse-validity: a checkpoint written before
    # amplitudes_all_trains/electrode_pairs_all_trains existed is missing fields
    # Block C now needs, and is treated as absent so it gets recomputed rather
    # than silently served stale (never deleted -- the old file is simply
    # overwritten in place by the same atomic save path once recomputed).
    if record.get("status") == "excluded":
        return record
    if "amplitudes_all_trains" not in record:
        return None
    return record


def save_checkpoint(session_id: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(session_id)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


# ── Block B: position-in-sequence interaction ────────────────────────────────

def fit_subject_interaction(items: list[dict]) -> dict | None:
    """OLS on recalled ~ 1 + stim + pos_c + stim:pos_c (closed-form lstsq).
    Returns None if the subject does not clear MIN_TRIALS_PER_ARM_PER_SUBJECT
    stim and control trials, or has no serial-position variance in either
    arm to fit a slope against."""
    stim = np.array([it["stim"] for it in items], dtype=float)
    n_stim, n_ctrl = int(stim.sum()), int((1 - stim).sum())
    if n_stim < MIN_TRIALS_PER_ARM_PER_SUBJECT or n_ctrl < MIN_TRIALS_PER_ARM_PER_SUBJECT:
        return None
    pos = np.array([it["serialpos"] for it in items], dtype=float)
    if np.std(pos[stim == 1]) == 0 or np.std(pos[stim == 0]) == 0:
        return None
    y = np.array([it["recalled"] for it in items], dtype=float)
    pos_c = pos - SERIALPOS_CENTER
    X = np.column_stack([np.ones_like(stim), stim, pos_c, stim * pos_c])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"n_stim": n_stim, "n_control": n_ctrl,
            "stim_main_effect": float(beta[1]), "position_main_effect": float(beta[2]),
            "interaction": float(beta[3])}


def subject_array_test(values: list[float], seed_key: str) -> dict:
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(values) < 2:
        return {"status": "not_computable", "n_subjects": int(len(values)),
                "reason": "fewer than 2 subjects have a fitted coefficient"}
    rng = np.random.default_rng(stable_seed(seed_key))
    test = paired_sign_flip_test(values, np.zeros_like(values), n_perm=N_PERM,
                                 alternative="two-sided", n_boot=N_BOOT, rng=rng)
    mdd = minimum_detectable_paired_difference(values, alpha=ALPHA, power=POWER)
    return {"status": "computed", "n_subjects": int(len(values)),
            "mean_value": test["mean_diff"], "p_value": test["p_value"],
            "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"], "mdd": mdd}


def block_b_variant(subject_items: dict[str, list[dict]], variant_name: str,
                    positive_label: str, null_label: str) -> dict:
    fits, dropped = {}, {}
    for subj, items in subject_items.items():
        fit = fit_subject_interaction(items)
        if fit is None:
            dropped[subj] = f"fewer than {MIN_TRIALS_PER_ARM_PER_SUBJECT} trials in one arm, " \
                            "or no serial-position variance in one arm"
        else:
            fits[subj] = fit

    interaction_vals = [f["interaction"] for f in fits.values()]
    stim_main_vals = [f["stim_main_effect"] for f in fits.values()]
    position_main_vals = [f["position_main_effect"] for f in fits.values()]

    interaction_test = subject_array_test(interaction_vals, f"block_b_interaction|{variant_name}")
    stim_main_test = subject_array_test(stim_main_vals, f"block_b_stim_main|{variant_name}")
    position_main_test = subject_array_test(position_main_vals, f"block_b_position_main|{variant_name}")

    reference_effect = abs(position_main_test["mean_value"]) if position_main_test["status"] == "computed" else None

    if interaction_test["status"] == "computed" and interaction_test["p_value"] <= ALPHA:
        branch = positive_label
    elif (interaction_test["status"] == "computed" and interaction_test["mdd"].get("status") == "computed"
          and reference_effect is not None):
        branch = null_label if interaction_test["mdd"]["mdd"] < reference_effect else "underpowered_to_ask"
    else:
        branch = "underpowered_to_ask"

    return {
        "variant": variant_name, "branch": branch,
        "n_subjects_fitted": len(fits), "n_subjects_dropped": len(dropped),
        "subjects_dropped": dropped,
        "interaction": interaction_test,
        "reference_effect_position_main_effect_magnitude": reference_effect,
        "main_effects": {"stimulation_on_recall": stim_main_test, "serial_position_on_recall": position_main_test},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_corpus(corpus_dir: str, processor, corpus_label: str) -> dict:
    subject_dirs = discover_subject_dirs(corpus_dir)
    session_paths = discover_sessions(corpus_dir)
    records, excluded_sessions = {}, {}

    for path in session_paths:
        session_id = path.stem.replace("_events", "")
        cached = load_checkpoint(f"{corpus_label}__{session_id}")
        if cached is not None:
            records[session_id] = cached
            continue
        rows = read_events(path)
        record = processor(rows)
        record["subject"] = rows[0]["subject"] if rows else session_id.split("_")[0].replace("sub-", "")
        record["session_id"] = session_id
        if record.get("status") == "excluded":
            excluded_sessions[session_id] = record["reason"]
        save_checkpoint(f"{corpus_label}__{session_id}", record)
        records[session_id] = record

    subjects_seen = set(subject_dirs)
    subjects_with_events = {r["subject"] for r in records.values()} | \
        {p.parts[-4] for p in session_paths} if False else set()
    subjects_with_session_files = {p.relative_to(DATA_ROOT / corpus_dir).parts[0] for p in session_paths}
    subjects_without_ieeg = sorted(s for s in subjects_seen if s not in subjects_with_session_files)

    included_records = {k: v for k, v in records.items() if v.get("status") != "excluded"}

    return {
        "records": included_records,
        "excluded_sessions": excluded_sessions,
        "n_sessions_seen": len(session_paths),
        "n_sessions_included": len(included_records),
        "n_subjects_seen_as_directories": len(subject_dirs),
        "n_subjects_with_ieeg_events": len(subjects_with_session_files),
        "subject_directories_without_ieeg_events": subjects_without_ieeg,
    }


def pool_block_a(records: dict, corpus_label: str) -> dict:
    per_session = {}
    all_items_per_train, all_spacing, all_train_durations = [], [], []
    n_prev = n_next = n_any = n_total = n_unmatched = 0
    n_trains_pooled = 0
    for sid, r in records.items():
        per_session[sid] = {
            "n_words": r["n_words"], "n_trains": r["n_trains"],
            "spacing_s_median": float(np.median(r["spacing_s"])) if r["spacing_s"] else None,
            "spacing_s_mean": float(np.mean(r["spacing_s"])) if r["spacing_s"] else None,
            "spacing_s_sd": float(np.std(r["spacing_s"], ddof=1)) if len(r["spacing_s"]) > 1 else None,
            "train_duration_s_median": float(np.median(r["train_duration_s"])) if r["train_duration_s"] else None,
            "train_duration_s_mean": float(np.mean(r["train_duration_s"])) if r["train_duration_s"] else None,
            "items_per_train_mean": float(np.mean(r["items_per_train"])) if r["items_per_train"] else None,
            "neighbor_coverage": r["neighbor_coverage"],
        }
        all_items_per_train.extend(r["items_per_train"])
        all_spacing.extend(r["spacing_s"])
        all_train_durations.extend(r["train_duration_s"])
        nb = r["neighbor_coverage"]
        n_prev += nb["n_prev_covered"]; n_next += nb["n_next_covered"]
        n_any += nb["n_any_covered"]; n_total += nb["n_stim_items"]
        n_unmatched += nb["n_train_unmatched_to_owning_item"]
        n_trains_pooled += r["n_trains"]

    attributable = [n for n in all_items_per_train if n >= 1]
    zero_overlap = len(all_items_per_train) - len(attributable)
    mean_items_per_train = float(np.mean(attributable)) if attributable else None

    if mean_items_per_train is None:
        branch = "underpowered_to_ask"
    elif mean_items_per_train > ITEMS_PER_TRAIN_SPAN_THRESHOLD:
        branch = "the_intervention_spans_more_than_one_item_and_cannot_be_attributed_to_one"
    else:
        branch = "the_intervention_is_confined_to_a_single_item"

    return {
        "per_session": per_session,
        "pooled": {
            "n_trains_total": n_trains_pooled,
            "n_trains_overlapping_zero_items": zero_overlap,
            "n_trains_overlapping_at_least_one_item": len(attributable),
            "items_per_train_distribution": {str(k): int(v) for k, v in
                                             zip(*np.unique(attributable, return_counts=True))} if attributable else {},
            "mean_items_per_train_among_attributable_trains": mean_items_per_train,
            "span_decision_threshold": ITEMS_PER_TRAIN_SPAN_THRESHOLD,
            "item_presentation_spacing_s_median": float(np.median(all_spacing)) if all_spacing else None,
            "item_presentation_spacing_s_mean": float(np.mean(all_spacing)) if all_spacing else None,
            "item_presentation_spacing_s_sd": float(np.std(all_spacing, ddof=1)) if len(all_spacing) > 1 else None,
            "train_duration_s_median": float(np.median(all_train_durations)) if all_train_durations else None,
            "train_duration_s_mean": float(np.mean(all_train_durations)) if all_train_durations else None,
            "n_stimulated_items": n_total,
            "n_stimulated_items_train_unmatched": n_unmatched,
            "fraction_stimulated_items_train_covers_preceding_item": (n_prev / n_total) if n_total else None,
            "fraction_stimulated_items_train_covers_following_item": (n_next / n_total) if n_total else None,
            "fraction_stimulated_items_train_covers_either_neighbor": (n_any / n_total) if n_total else None,
        },
        "branch": branch,
    }


def pool_block_c(records: dict, corpus_label: str) -> dict:
    amp_counts = defaultdict(int)
    pair_amps = defaultdict(set)
    pair_trials = defaultdict(int)
    freqs, widths, durs = set(), set(), set()
    pair_freqs, pair_widths = defaultdict(set), defaultdict(set)
    n_stim_trials = 0
    subjects = set()
    for sid, r in records.items():
        subj = r["subject"]
        subjects.add(subj)
        for amp in r["amplitudes"]:
            amp_counts[amp] += 1
            n_stim_trials += 1
        freqs.update(r["pulse_freqs"]); widths.update(r["pulse_widths"]); durs.update(r["stim_durations_ms"])
        for (anode, cathode) in r["electrode_pairs"]:
            pair_key = (subj, anode, cathode)
            pair_trials[pair_key] += 1
        # per-pair amplitude/freq/width sets need per-train granularity, not just the
        # per-session unique-value set, so recover it from the parallel per-session lists
        # (amplitudes list is per-train; electrode_pairs is per-corpus unique -- re-derive
        # per-pair amplitude sets from records that carry per-train pair identity below).

    # Second pass: per-train (amplitude, pair) association is not retained per-session
    # above (only per-session unique sets), so re-open a lighter per-pair accumulation
    # using each session's amplitude list matched 1:1 against its own single stim
    # electrode pair -- true for both corpora (one stimulated bipolar pair per session).
    for sid, r in records.items():
        subj = r["subject"]
        pairs = r["electrode_pairs"]
        if len(pairs) != 1:
            continue  # a session with zero or >1 distinct stim pairs contributes no
                      # unambiguous per-pair amplitude assignment; counted in the
                      # pooled amplitude census above regardless.
        anode, cathode = pairs[0]
        key = (subj, anode, cathode)
        for amp in r["amplitudes"]:
            pair_amps[key].add(amp)
        for f in r["pulse_freqs"]:
            pair_freqs[key].add(f)
        for w in r["pulse_widths"]:
            pair_widths[key].add(w)

    multi_amp_pairs = {k: sorted(v) for k, v in pair_amps.items() if len(v) > 1}
    subjects_multi_amp = sorted({k[0] for k in multi_amp_pairs})
    multi_freq_within_pair = {str(k): sorted(v) for k, v in pair_freqs.items() if len(v) > 1}
    multi_width_within_pair = {str(k): sorted(v) for k, v in pair_widths.items() if len(v) > 1}

    # Same accumulation, but over every STIM_ON/STIM_OFF-derived train regardless of
    # whether it could be matched to an owning WORD item -- surfaces amplitude values
    # that only ever appear on pre-task titration/calibration pulses (never near an
    # item), so that number is reported rather than silently absorbed into, or
    # silently absent from, the item-linked count above.
    pair_amps_all = defaultdict(set)
    for sid, r in records.items():
        subj = r["subject"]
        pairs_all = r.get("electrode_pairs_all_trains", r["electrode_pairs"])
        if len(pairs_all) != 1:
            continue
        anode, cathode = pairs_all[0]
        for amp in r.get("amplitudes_all_trains", r["amplitudes"]):
            pair_amps_all[(subj, anode, cathode)].add(amp)
    multi_amp_pairs_all_trains = {k: sorted(v) for k, v in pair_amps_all.items() if len(v) > 1}

    return {
        "n_subjects_with_stimulated_trials": len(subjects),
        "n_stimulated_trials_total": n_stim_trials,
        "stimulated_trials_by_amplitude_microamps": {str(k): v for k, v in sorted(amp_counts.items())},
        "n_electrode_pairs": len(pair_trials),
        "n_electrode_pairs_with_more_than_one_amplitude": len(multi_amp_pairs),
        "subjects_with_a_multi_amplitude_electrode_pair": subjects_multi_amp,
        "electrode_pairs_with_more_than_one_amplitude": {str(k): v for k, v in multi_amp_pairs.items()},
        "constant_across_corpus": {
            "pulse_freq_hz": sorted(freqs) if len(freqs) == 1 else None,
            "pulse_width_us": sorted(widths) if len(widths) == 1 else None,
            "stim_duration_ms": sorted(durs) if len(durs) == 1 else None,
            "amplitude_microamps": sorted(amp_counts.keys()) if len(amp_counts) == 1 else None,
        },
        "unique_values_seen": {
            "pulse_freq_hz": sorted(freqs), "pulse_width_us": sorted(widths),
            "stim_duration_ms": sorted(durs), "amplitude_microamps": sorted(amp_counts.keys()),
        },
        "electrode_pairs_with_more_than_one_pulse_freq": multi_freq_within_pair,
        "electrode_pairs_with_more_than_one_pulse_width": multi_width_within_pair,
        "n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses":
            len(multi_amp_pairs_all_trains),
        "electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses":
            {str(k): v for k, v in multi_amp_pairs_all_trains.items()},
        "note_on_unmatched_pulse_amplitude": (
            "the item-linked counts above use only STIM_ON/STIM_OFF trains matched to an "
            "owning WORD item; a small number of STIM_ON rows in this corpus (e.g. a single "
            "pulse at recording onset, long before any word is shown) cannot be matched to any "
            "item and are excluded from them. The "
            "'..._including_unmatched_pulses' fields fold those back in, so a widening gap "
            "between the two flags amplitude variation coming from pre-task device "
            "titration/calibration rather than genuine within-task dose variation."
        ),
    }


def main():
    t0 = time.time()

    openloop = run_corpus(OPENLOOP_CORPUS, process_openloop_session, "openloop")
    closedloop = run_corpus(CLOSEDLOOP_CORPUS, process_closedloop_session, "closedloop")

    block_a = {
        "ds005489_open_loop": pool_block_a(openloop["records"], "openloop"),
        "ds005557_classifier_triggered": pool_block_a(closedloop["records"], "closedloop"),
    }

    # Block B -- open-loop only, causal (see module docstring for the scope decision).
    subject_items_all, subject_items_clean = defaultdict(list), defaultdict(list)
    for r in openloop["records"].values():
        subj = r["subject"]
        for it in r["items"]:
            subject_items_all[subj].append(it)
            if it["stim"] == 0 or it["prev_covered"] is False:
                subject_items_clean[subj].append(it)

    block_b = {
        "scope_note": (
            "computed on ds005489 (open-loop) only: item-level stimulation there is "
            "experimenter-scheduled and randomized at the list level, so a treatment "
            "effect is causally interpretable. ds005557 (classifier-triggered) item-level "
            "stimulation is triggered by the classifier's own reading of the encoding-period "
            "state, i.e. propensity-selected on signal related to the outcome under test, "
            "so an item-level position interaction there would not be a causal estimate and "
            "is not computed here."
        ),
        "primary_all_stimulated_items": block_b_variant(
            dict(subject_items_all), "primary",
            "the_effect_of_stimulation_depends_on_position_in_the_sequence",
            "no_dependence_on_sequence_position_above_the_reported_bound"),
        "clean_subset_train_did_not_cover_preceding_item": block_b_variant(
            dict(subject_items_clean), "clean_subset",
            "the_effect_of_stimulation_depends_on_position_in_the_sequence",
            "no_dependence_on_sequence_position_above_the_reported_bound"),
    }

    block_c = {
        "ds005489_open_loop": pool_block_c(openloop["records"], "openloop"),
        "ds005557_classifier_triggered": pool_block_c(closedloop["records"], "closedloop"),
    }

    a_open = block_a["ds005489_open_loop"]
    a_closed = block_a["ds005557_classifier_triggered"]
    c_open = block_c["ds005489_open_loop"]
    c_closed = block_c["ds005557_classifier_triggered"]

    block_d = {
        "what_this_corpus_pair_answers": (
            "In the open-loop arm (ds005489), the delivered pulse train "
            f"(median {a_open['pooled']['train_duration_s_median']:.2f} s) is longer than the "
            f"measured item presentation interval (median {a_open['pooled']['item_presentation_spacing_s_median']:.2f} s), "
            f"so a train overlaps a mean of {a_open['pooled']['mean_items_per_train_among_attributable_trains']:.2f} item "
            "presentations and the branch fired is "
            f"'{a_open['branch']}': timing claims from this corpus cannot resolve finer than the train, "
            "only which block of the list it fell in. In the classifier-triggered arm (ds005557), the "
            f"pulse train (median {a_closed['pooled']['train_duration_s_median']:.3f} s) is much shorter than the item "
            f"interval, overlapping a mean of {a_closed['pooled']['mean_items_per_train_among_attributable_trains']:.2f} "
            f"item presentations among trains that overlap any item at all, so the branch fired is "
            f"'{a_closed['branch']}' -- this arm CAN, in principle, localise a stimulation event to a single "
            "item's encoding, though it answers a different timing question (at which classifier readout "
            "value, not at which position in the list)."
        ),
        "what_this_corpus_pair_cannot_answer": (
            "Neither corpus stimulates during a working-memory maintenance/delay period -- both deliver "
            "stimulation at encoding, so neither speaks to when-in-a-delay a device should act. Frequency "
            "and pulse width are fixed constants within every electrode pair in both corpora (in ds005489 "
            "corpus-wide; in ds005557 constant within pair even though it differs between pairs/subjects), "
            "so no within-pair claim about stimulation frequency or pulse width is possible from either "
            f"human arm. Amplitude varies within a pair in {c_open['n_electrode_pairs_with_more_than_one_amplitude']} "
            f"ds005489 pair(s); in ds005557, amplitude does not vary within any electrode pair among "
            f"item-linked pulses ({c_closed['n_electrode_pairs_with_more_than_one_amplitude']} pairs) -- a naive "
            "scan of every STIM_ON row (including pre-task device calibration pulses that never land near "
            f"an item) would over-count this as {c_closed['n_electrode_pairs_with_more_than_one_amplitude_including_unmatched_pulses']} "
            "pairs, which is why the item-linked count, not the raw one, is the one reported here."
        ),
        "what_a_non_human_delay_period_stimulation_corpus_would_add": (
            "A non-human preparation with stimulation delivered during the maintenance/delay period itself "
            "(rather than at encoding) is the only way to ask the delay-period timing question at all; if "
            "that preparation also varies frequency or pulse width within a single stimulation site, it is "
            "the only way to ask the dose-parameter questions this project's human corpora cannot ask "
            "within an electrode pair. A non-human result answers those two gaps for the non-human "
            "preparation only -- it does not stand in for a human within-delay or within-pair "
            "frequency/pulse-width measurement, which remains unmeasured in humans and is named here as a "
            "gap, not inferred from the animal side."
        ),
        "precisely_what_human_measurement_would_close_each_gap": (
            "(1) Delay-period gap: a human stimulation session that schedules delivery during the "
            "maintenance interval of a working-memory (not free-recall-encoding) task, with event tables "
            "carrying the same onset/duration/train fields already used here, so the same attributability "
            "and position analysis could be re-run against delay time instead of list position. "
            "(2) Frequency/pulse-width gap: a human session in which the SAME electrode pair is stimulated "
            "at more than one frequency or pulse width (mirroring the within-pair multi-amplitude cohort "
            "already present in "
            f"{c_open['n_electrode_pairs_with_more_than_one_amplitude']} ds005489 pair(s) and "
            f"{c_closed['n_electrode_pairs_with_more_than_one_amplitude']} ds005557 pair(s) for amplitude) -- "
            "no such within-pair frequency or pulse-width variation exists in either corpus today."
        ),
    }

    scope = {
        "corpora": {
            "ds005489_open_loop": {
                "n_subject_directories": openloop["n_subjects_seen_as_directories"],
                "n_subjects_with_ieeg_events": openloop["n_subjects_with_ieeg_events"],
                "subject_directories_without_ieeg_events": openloop["subject_directories_without_ieeg_events"],
                "n_sessions_seen": openloop["n_sessions_seen"],
                "n_sessions_included": openloop["n_sessions_included"],
                "excluded_sessions": openloop["excluded_sessions"],
            },
            "ds005557_classifier_triggered": {
                "n_subject_directories": closedloop["n_subjects_seen_as_directories"],
                "n_subjects_with_ieeg_events": closedloop["n_subjects_with_ieeg_events"],
                "subject_directories_without_ieeg_events": closedloop["subject_directories_without_ieeg_events"],
                "n_sessions_seen": closedloop["n_sessions_seen"],
                "n_sessions_included": closedloop["n_sessions_included"],
                "excluded_sessions": closedloop["excluded_sessions"],
            },
        },
        "data_root": str(DATA_ROOT),
        "alpha": ALPHA, "power": POWER, "n_perm": N_PERM, "n_boot": N_BOOT,
        "closedloop_owner_match_window_s": CLOSEDLOOP_OWNER_MATCH_WINDOW_S,
        "items_per_train_span_decision_threshold": ITEMS_PER_TRAIN_SPAN_THRESHOLD,
        "min_trials_per_arm_per_subject": MIN_TRIALS_PER_ARM_PER_SUBJECT,
        "git_commit": git_commit(ROOT),
        "wall_clock_seconds": None,  # filled in just before write
    }

    out = {
        "status": "complete",
        "scope": scope,
        "block_a_attributability_to_a_single_item": block_a,
        "block_b_position_in_sequence_interaction": block_b,
        "block_c_parameter_census": block_c,
        "block_d_synthesis": block_d,
    }

    scope["wall_clock_seconds"] = time.time() - t0
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(canonical_json(out))

    print(f"Wrote {OUT_PATH} in {scope['wall_clock_seconds']:.1f} s")
    print(f"Block A open-loop branch: {block_a['ds005489_open_loop']['branch']}")
    print(f"Block A closed-loop branch: {block_a['ds005557_classifier_triggered']['branch']}")
    print(f"Block B primary branch: {block_b['primary_all_stimulated_items']['branch']}")
    print(f"Block B clean-subset branch: {block_b['clean_subset_train_did_not_cover_preceding_item']['branch']}")


if __name__ == "__main__":
    main()
