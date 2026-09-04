#!/usr/bin/env python3
"""Does distance from the stimulated site to the recording contacts that
carry the stimulation-response component, measured before any stimulation
trial is looked at, predict how far that component moves and how much
recall changes when the site is driven?

Two human intracranial free-recall stimulation corpora ship per-contact
anatomy in their release (MNI coordinates, a shank/lead identifier, and
three independent anatomical labelling schemes) alongside a per-session
component-displacement estimate already computed elsewhere
(results/human_stimulation_component_response.json) and, for one corpus, a
per-session behavioural stimulated-vs-control recall difference
(results/causal_ram.json, open-loop; results/causal_ram_closedloop.json,
classifier-triggered). This module recomputes neither: it reads both, joins
them session by session, and adds exactly one new session-level quantity --
a geometric predictor built from each session's own non-stimulated trials.

The predictor: for a session and a choice of which recording channels are
trusted, take the fixed reference direction that channel set's non-
stimulated trials define (unmodified from the displacement module -- the
mean unit-normalised trial vector, no leave-one-out), the per-channel
carriage of that direction (the absolute value of each channel's entry),
and the Euclidean distance in MNI space from the midpoint of the driven
bipolar pair to the midpoint of each trusted recording channel's own two
contacts. The predictor is the carriage-weighted mean of those distances --
a single number in millimetres: how far, on average, weighted by how
strongly each channel carries the pre-stimulation component, the driven
site sits from the tissue that carries it. A smaller value means the site
sits closer to the carrying tissue.

Channels nearest the driven pair carry the largest stimulation artifact, and
an artifact gradient would produce exactly the same distance relationship
this analysis tests. So every distance relationship here is computed three
ways over the same channel sets the displacement module already defines --
every recording channel, the driven bipolar pair excluded, and the entire
shank carrying the driven pair excluded -- and a relationship visible only
with the near channels included is reported as not separable from
stimulation artifact, never as a targeting result.

Anatomical region is a coverage-confounded, purely observational grouping
here (electrode placement is clinically chosen, not randomised) and is kept
separate from the geometric analysis; it never produces a causal region
claim.

Outputs:
  results/stimulation_site_targeting_map.json

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python \
        scripts/run_stimulation_site_targeting_map.py [--smoke N]
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import csv
import json
import re
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from provenance import canonical_json, git_commit  # noqa: E402
from statistics import fdr_bh  # noqa: E402
from run_human_stimulation_component_response import (  # noqa: E402
    ALPHA,
    MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    _bin_averaged,
    _reference_direction,
    channel_condition_masks,
    load_raw_features,
    subject_aggregated_correlation,
    subject_clustered_mean_test,
)
from run_ram_openloop_pipeline import DATA as OPENLOOP_DATA  # noqa: E402
from run_ram_closedloop_pipeline import DATA as CLOSEDLOOP_DATA  # noqa: E402

RESULTS = ROOT / "results"
OUTPUT_PATH = RESULTS / "stimulation_site_targeting_map.json"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_stimulation_site_targeting_map"

COMPONENT_RESPONSE_PATH = RESULTS / "human_stimulation_component_response.json"
CAUSAL_ARTIFACT = {
    "open_loop_ds005489": RESULTS / "causal_ram.json",
    "closed_loop_ds005557": RESULTS / "causal_ram_closedloop.json",
}
CORPUS_IS_CAUSAL = {"open_loop_ds005489": True, "closed_loop_ds005557": False}
CORPUS_DATA_DIR = {"open_loop_ds005489": OPENLOOP_DATA, "closed_loop_ds005557": CLOSEDLOOP_DATA}

CONDITION_NAMES = ["full_channel_set", "excluding_stimulated_pair", "excluding_stimulated_shank"]
LABEL_SCHEMES = ["ind.region", "das.region", "stein.region"]

MIN_CONTROL_TRIALS = 8            # same floor the displacement module applies to its own control pool
MIN_CHANNELS_WITH_KNOWN_GEOMETRY = 3  # fewest weighted points a carriage-weighted mean distance is trusted with
MIN_SUBJECTS_PER_REGION_CELL = 4

MISSING_NUMERIC_TOKENS = {"", "n/a", "-999", "nan"}

PREDICTOR_DEFINITION = (
    "Per session and per trusted-channel condition: the fixed reference direction that "
    "condition's non-stimulated trials define (mean unit-normalised trial vector over the "
    "trusted channels, no leave-one-out -- the same reference the displacement module scores "
    "stimulated trials against), the per-channel carriage of that direction (absolute value of "
    "each channel's entry), and the Euclidean distance in MNI152NLin6ASym space from the "
    "midpoint of the driven bipolar pair's two contacts to the midpoint of each trusted "
    "recording channel's own two contacts. The predictor is "
    "sum(carriage_c * distance_c) / sum(carriage_c) over channels with both a finite carriage "
    "value and a resolvable location -- the carriage-weighted mean distance, in millimetres, "
    "from the driven site to the tissue that carries the component. Smaller means closer to the "
    "carrying tissue. The outcome each predictor is tested against is the absolute value of the "
    "normalised displacement (a magnitude, since displacement is signed and 'larger' means "
    "further from baseline in either direction) and the absolute value of the stimulated-minus-"
    "control recall difference."
)

# Standard cortical-lobe grouping (frontal/temporal/parietal/occipital/cingulate-limbic/insula),
# covering exactly the region-label vocabulary this release's three anatomical schemes use.
# das.region is a hippocampal-subfield-only scheme in this release (electrode-table sentinel
# audit: >99% of contacts read 'n/a' in that column) -- every non-missing value in it names a
# medial temporal lobe structure, so all of them map to "temporal".
_FRONTAL = {"superiorfrontal", "rostralmiddlefrontal", "caudalmiddlefrontal", "parsopercularis",
            "parstriangularis", "parsorbitalis", "lateralorbitofrontal", "medialorbitofrontal",
            "precentral", "paracentral", "frontalpole", "dlpfc"}
_TEMPORAL = {"superiortemporal", "middletemporal", "inferiortemporal", "bankssts", "fusiform",
             "transversetemporal", "temporalpole", "parahippocampal", "entorhinal",
             "ba35", "ba36", "ca1", "ca2", "ca3", "dg", "erc", "phc", "sub", "head", "amy",
             "mtl wm", "middle temporal gyrus", "prc"}
_PARIETAL = {"superiorparietal", "inferiorparietal", "supramarginal", "postcentral", "precuneus"}
_OCCIPITAL = {"lateraloccipital", "lingual", "cuneus", "pericalcarine"}
_CINGULATE_LIMBIC = {"caudalanteriorcingulate", "rostralanteriorcingulate", "posteriorcingulate",
                      "isthmuscingulate", "acg", "mcg", "pcg"}
_INSULA = {"insula"}
REGION_TO_LOBE = {}
for _lobe, _members in (("frontal", _FRONTAL), ("temporal", _TEMPORAL), ("parietal", _PARIETAL),
                        ("occipital", _OCCIPITAL), ("cingulate_limbic", _CINGULATE_LIMBIC),
                        ("insula", _INSULA)):
    for _m in _members:
        REGION_TO_LOBE[_m] = _lobe


def _lobe_for_label(scheme: str, raw_label: str) -> str:
    """das.region/ind.region labels are used as-is; stein.region carries a
    'Left '/'Right ' hemisphere prefix that lobe membership ignores."""
    base = raw_label
    if scheme == "stein.region":
        base = re.sub(r"^(Left|Right)\s+", "", raw_label)
    return REGION_TO_LOBE.get(base.strip().lower(), "unmapped_lobe")


# ── Checkpointing (fit-level, crash-proof; mirrors the displacement module) ────

def _checkpoint_path(unit: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", unit)
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_checkpoint(unit: str, record: dict) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(unit: str, fit_fn):
    cached = load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(unit, record)
    return record


# ── Electrode-table anatomy ─────────────────────────────────────────────────

def _numeric_or_nan(value) -> float:
    """-999 and 'n/a' both appear as missing-value sentinels for numeric
    fields in this release's electrode tables; neither may enter a distance
    or an average."""
    if value is None:
        return float("nan")
    token = value.strip().lower()
    if token in MISSING_NUMERIC_TOKENS:
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def _electrode_table_paths(data_dir: Path, session_key: str) -> tuple[Path, Path]:
    ieeg_json = data_dir / session_key
    stem = ieeg_json.name.replace("_acq-bipolar_ieeg.json", "")
    mni_path = ieeg_json.parent / f"{stem}_space-MNI152NLin6ASym_electrodes.tsv"
    tal_path = ieeg_json.parent / f"{stem}_space-Talairach_electrodes.tsv"
    return mni_path, tal_path


def load_electrode_table(data_dir: Path, session_key: str) -> dict:
    """MNI coordinates are required for the geometric distance analysis and
    are read only from the MNI152NLin6ASym-space table. The three
    anatomical labelling schemes are shipped in both space variants of the
    table; if the MNI table is absent for a session (a real gap in this
    release), the region labels are still read from the Talairach-space
    table so the anatomical map is not forced to drop that session too --
    the space actually used is recorded on every session."""
    mni_path, tal_path = _electrode_table_paths(data_dir, session_key)
    coords: dict[str, tuple[float, float, float]] = {}
    labels: dict[str, dict[str, str]] = {}
    space_for_labels = None
    table_path = mni_path if mni_path.exists() else (tal_path if tal_path.exists() else None)
    if table_path is not None:
        with open(table_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            labels[name] = {scheme: (row.get(scheme) or "n/a").strip() or "n/a" for scheme in LABEL_SCHEMES}
        space_for_labels = "MNI152NLin6ASym" if table_path == mni_path else "Talairach"
    if mni_path.exists():
        with open(mni_path) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            x, y, z = _numeric_or_nan(row.get("x")), _numeric_or_nan(row.get("y")), _numeric_or_nan(row.get("z"))
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                coords[name] = (x, y, z)
    return {"coords": coords, "labels": labels, "space_for_labels": space_for_labels,
            "mni_table_found": mni_path.exists()}


def channel_midpoint(channel_name: str, coords: dict) -> tuple[float, float, float] | None:
    """A recording channel is a bipolar pair 'A-B'; its location is the
    midpoint of its two contacts, the same convention used for the driven
    pair's own site location."""
    parts = channel_name.split("-")
    if len(parts) != 2:
        return None
    a, b = parts
    if a not in coords or b not in coords:
        return None
    pa, pb = np.array(coords[a]), np.array(coords[b])
    return tuple(((pa + pb) / 2.0).tolist())


def euclidean(p, q) -> float:
    return float(np.linalg.norm(np.array(p) - np.array(q)))


# ── Session enumeration and join ────────────────────────────────────────────

def _causal_key(session_key: str) -> str:
    return session_key.replace("_ieeg.json", "_ieeg.edf")


def load_admitted_sessions() -> list[dict]:
    """Every session block_b (the delivered displacement artifact) admitted,
    with its anode/cathode/channel-condition displacement carried over
    unchanged and a behavioural row joined in from the matching corpus's
    causal artifact. A session block_b admitted but this join cannot match
    is still returned, tagged with a machine-readable behaviour-join
    failure reason -- it contributes to Block A's geometry/displacement
    relationship and to Block B's anatomical map even without behaviour."""
    component = json.loads(COMPONENT_RESPONSE_PATH.read_text())
    block_b = component["block_b"]["per_session"]

    causal_tables = {name: json.loads(path.read_text())["per_session"] for name, path in CAUSAL_ARTIFACT.items()}

    sessions = []
    for session_key, rec in block_b.items():
        corpus = rec["corpus"]
        causal_table = causal_tables.get(corpus, {})
        causal_row = causal_table.get(_causal_key(session_key))
        if causal_row is not None:
            behavior = {
                "status": "computed",
                "recall_rate_stim": causal_row["recall_rate_stim"],
                "recall_rate_ctrl": causal_row["recall_rate_ctrl"],
                "recall_diff": causal_row["recall_rate_stim"] - causal_row["recall_rate_ctrl"],
                "n_words": causal_row["n_words"],
            }
        else:
            behavior = {"status": "excluded", "reason": "no_behavioral_outcome_in_delivered_causal_artifact"}
        sessions.append({
            "session_key": session_key, "corpus": corpus, "subject": rec["subject"],
            "anode": rec["anode"], "cathode": rec["cathode"], "stim_channel": rec["stim_channel"],
            "displacement_conditions": rec.get("conditions", {}),
            "behavior": behavior,
        })
    return sessions


# ── Per-session geometric predictor ─────────────────────────────────────────

def predictor_for_channel_condition(ctrl_activity: np.ndarray, kept_names: list[str], coords: dict,
                                     stim_site: tuple[float, float, float], n_channels_in_mask: int) -> dict:
    """The carriage-weighted mean distance for one session and one trusted-
    channel condition, given that condition's own non-stimulated-trial
    activity matrix and channel names -- factored out of
    targeting_predictor_session so the three-way channel control (a
    different `kept_names`/`ctrl_activity` per condition, same formula) is
    directly testable without a cached raw-feature file."""
    if ctrl_activity.shape[0] < MIN_CONTROL_TRIALS:
        return {"status": "too_few_control_trials", "n_channels": n_channels_in_mask,
                "n_control_trials": int(ctrl_activity.shape[0])}
    reference_direction = _reference_direction(ctrl_activity)
    if not np.all(np.isfinite(reference_direction)):
        return {"status": "degenerate_reference_direction", "n_channels": n_channels_in_mask}
    carriage = np.abs(reference_direction)
    distances = np.full(len(kept_names), np.nan)
    for i, name in enumerate(kept_names):
        midpoint = channel_midpoint(name, coords)
        if midpoint is not None:
            distances[i] = euclidean(midpoint, stim_site)
    valid = np.isfinite(distances) & np.isfinite(carriage)
    n_valid = int(valid.sum())
    if n_valid < MIN_CHANNELS_WITH_KNOWN_GEOMETRY or float(carriage[valid].sum()) <= 0.0:
        return {"status": "insufficient_channel_geometry", "n_channels": n_channels_in_mask,
                "n_channels_with_known_geometry": n_valid}
    weights = carriage[valid]
    dists = distances[valid]
    predictor = float(np.sum(weights * dists) / np.sum(weights))
    return {
        "status": "computed", "n_channels": n_channels_in_mask,
        "n_channels_with_known_geometry": n_valid,
        "n_channels_missing_geometry": int(len(kept_names) - n_valid),
        "distance_weighted_carriage_predictor_mm": predictor,
        "unweighted_mean_distance_mm": float(np.mean(dists)),
    }


def _targeting_predictor_session(session_key: str, corpus: str, anode: str, cathode: str,
                                  stim_channel: str) -> dict:
    """The one new per-session computation this module adds: the carriage-
    weighted mean distance from the driven site to the recording channels,
    for each of the three trusted-channel conditions."""
    data_dir = CORPUS_DATA_DIR[corpus]
    electrode = load_electrode_table(data_dir, session_key)
    if not electrode["mni_table_found"]:
        return {"status": "excluded", "reason": "no_mni152_electrode_table",
                "space_for_labels": electrode["space_for_labels"]}
    coords = electrode["coords"]
    if anode not in coords or cathode not in coords:
        return {"status": "excluded", "reason": "missing_mni_coordinates_for_stimulated_pair"}
    stim_site = tuple(((np.array(coords[anode]) + np.array(coords[cathode])) / 2.0).tolist())

    raw = load_raw_features(f"{corpus}__{session_key}")
    if raw is None:
        return {"status": "excluded", "reason": "raw_epoched_features_not_cached"}
    ch_names = [str(c) for c in raw["ch_names"].tolist()]
    stim_flag = raw["stim_flag"]
    masks = channel_condition_masks(ch_names, anode, cathode, stim_channel)

    conditions = {}
    for cond, mask in masks.items():
        ctrl_activity = _bin_averaged(raw, mask)[stim_flag == 0]
        kept_names = [n for n, keep in zip(ch_names, mask) if keep]
        conditions[cond] = predictor_for_channel_condition(ctrl_activity, kept_names, coords, stim_site,
                                                             int(mask.sum()))
    return {"status": "computed", "stim_site_mni": list(stim_site), "conditions": conditions,
            "space_for_labels": electrode["space_for_labels"]}


def site_labels_for_session(session_key: str, corpus: str, anode: str, cathode: str) -> dict:
    """The stimulated pair's anatomical label under each of the three
    shipped schemes. The anode is the pre-declared representative contact
    for a session's site label (chosen before any result was read, since a
    single categorical label is needed to place a session in one region
    cell); anode/cathode agreement is recorded but never resolved by
    picking whichever contact makes a cell bigger."""
    data_dir = CORPUS_DATA_DIR[corpus]
    electrode = load_electrode_table(data_dir, session_key)
    if electrode["space_for_labels"] is None:
        return {"status": "excluded", "reason": "no_electrode_table_of_either_space"}
    out = {}
    for scheme in LABEL_SCHEMES:
        anode_label = electrode["labels"].get(anode, {}).get(scheme, "n/a")
        cathode_label = electrode["labels"].get(cathode, {}).get(scheme, "n/a")
        out[scheme] = {
            "anode_label": anode_label, "cathode_label": cathode_label,
            "anode_cathode_agree": anode_label == cathode_label,
            "site_label": anode_label,
            "lobe": _lobe_for_label(scheme, anode_label) if anode_label != "n/a" else "n/a",
        }
    return {"status": "computed", "space_used": electrode["space_for_labels"], "schemes": out}


# ── Subject-clustered classification (shared shape across every relationship) ──

def classify_targeting_relationship(corr_by_condition: dict) -> str:
    """Mirrors the displacement module's own three-way artifact-control
    classification: a relationship visible only with the near channels
    included is reported as not separable from stimulation artifact, and
    the headline branch requires it to survive the whole-shank exclusion
    with the same sign the unrestricted channel set shows."""
    full = corr_by_condition.get("full_channel_set", {})
    shank = corr_by_condition.get("excluding_stimulated_shank", {})
    if full.get("status") != "computed":
        return "underpowered_to_ask"
    full_significant = full["p_value"] <= ALPHA
    shank_significant = shank.get("status") == "computed" and shank["p_value"] <= ALPHA
    shank_same_sign = shank_significant and (shank["r"] < 0) == (full["r"] < 0)
    if full_significant and not (shank_significant and shank_same_sign):
        return "site_distance_effect_not_separable_from_recording_artifact"
    if shank_significant and shank_same_sign:
        return "stimulating_closer_to_the_carrying_contacts_moves_the_component_more"
    mdd_source = shank if shank.get("status") == "computed" else full
    mdd = mdd_source.get("mdd", {})
    if mdd.get("status") == "computed" and mdd["mdd"] < MEANINGFUL_EFFECT_THRESHOLD_R_UNITS:
        return "no_site_distance_relationship_above_the_reported_bound"
    return "underpowered_to_ask"


# ── Block A -- the geometric targeting relationship ─────────────────────────

def _geometry_rows(sessions: list[dict]) -> dict:
    """Runs (and checkpoints) the per-session geometric predictor for every
    admitted session, once, shared by Blocks A and C."""
    rows = {}
    for rec in sessions:
        key = f"geometry__{rec['corpus']}__{rec['session_key']}"
        result = run_checkpointed(key, lambda rec=rec: _targeting_predictor_session(
            rec["session_key"], rec["corpus"], rec["anode"], rec["cathode"], rec["stim_channel"]))
        rows[rec["session_key"]] = result
    return rows


def run_block_a(sessions: list[dict], geometry: dict) -> dict:
    """Open-loop (causal) corpus only; Block C repeats this in the
    classifier-triggered corpus. Two outcomes tested against the same
    per-condition geometric predictor: the delivered component displacement
    (all admitted sessions) and the delivered recall difference (only the
    subset with a behavioural join)."""
    per_session = {}
    displacement_by_condition = {c: {"predictor": [], "outcome": [], "subject": [], "session": []} for c in CONDITION_NAMES}
    behavior_by_condition = {c: {"predictor": [], "outcome": [], "subject": [], "session": []} for c in CONDITION_NAMES}

    for rec in sessions:
        geo = geometry.get(rec["session_key"], {"status": "excluded", "reason": "not_run"})
        per_session[rec["session_key"]] = {"subject": rec["subject"], "geometry": geo, "behavior": rec["behavior"]}
        if geo.get("status") != "computed":
            continue
        for cond in CONDITION_NAMES:
            geo_cond = geo["conditions"].get(cond, {})
            if geo_cond.get("status") != "computed":
                continue
            predictor = geo_cond["distance_weighted_carriage_predictor_mm"]

            disp_cond = rec["displacement_conditions"].get(cond, {})
            if disp_cond.get("status") == "computed" and disp_cond.get("normalised_displacement") is not None:
                displacement_by_condition[cond]["predictor"].append(predictor)
                displacement_by_condition[cond]["outcome"].append(abs(disp_cond["normalised_displacement"]))
                displacement_by_condition[cond]["subject"].append(rec["subject"])
                displacement_by_condition[cond]["session"].append(rec["session_key"])

            if rec["behavior"]["status"] == "computed":
                behavior_by_condition[cond]["predictor"].append(predictor)
                behavior_by_condition[cond]["outcome"].append(abs(rec["behavior"]["recall_diff"]))
                behavior_by_condition[cond]["subject"].append(rec["subject"])
                behavior_by_condition[cond]["session"].append(rec["session_key"])

    displacement_corr = {
        c: subject_aggregated_correlation(np.array(v["predictor"]), np.array(v["outcome"]), v["subject"])
        if v["predictor"] else {"status": "not_computable", "n_sessions": 0, "n_subjects": 0}
        for c, v in displacement_by_condition.items()
    }
    behavior_corr = {
        c: subject_aggregated_correlation(np.array(v["predictor"]), np.array(v["outcome"]), v["subject"])
        if v["predictor"] else {"status": "not_computable", "n_sessions": 0, "n_subjects": 0}
        for c, v in behavior_by_condition.items()
    }

    displacement_branch = classify_targeting_relationship(displacement_corr)
    behavior_branch = classify_targeting_relationship(behavior_corr)

    return {
        "per_session": per_session,
        "predictor_definition": PREDICTOR_DEFINITION,
        "displacement_relationship": {
            "by_channel_condition": displacement_corr,
            "n_sessions_by_condition": {c: len(v["session"]) for c, v in displacement_by_condition.items()},
            "n_subjects_by_condition": {c: len(set(v["subject"])) for c, v in displacement_by_condition.items()},
            "branch": displacement_branch,
        },
        "behavior_relationship": {
            "by_channel_condition": behavior_corr,
            "n_sessions_by_condition": {c: len(v["session"]) for c, v in behavior_by_condition.items()},
            "n_subjects_by_condition": {c: len(set(v["subject"])) for c, v in behavior_by_condition.items()},
            "branch": behavior_branch,
        },
        "branch": displacement_branch,
        "meaningful_effect_threshold_r_units": MEANINGFUL_EFFECT_THRESHOLD_R_UNITS,
    }


# ── Block B -- the anatomical map, reported as description ─────────────────

def run_block_b(sessions: list[dict], site_labels: dict) -> dict:
    """Descriptive and observational: coverage is clinical, no subject was
    randomised to a site, and this block never yields a causal region
    claim. Each of the three shipped labelling schemes is analysed and
    reported independently; where they disagree for a site that disagreement
    is visible in per_session, never resolved into one picked label."""
    schemes_out = {}
    for scheme in LABEL_SCHEMES:
        region_rows = defaultdict(list)   # region label -> list of (session_key, subject, |displacement|)
        lobe_rows = defaultdict(list)
        n_no_label = 0
        for rec in sessions:
            labels = site_labels.get(rec["session_key"], {})
            if labels.get("status") != "computed":
                n_no_label += 1
                continue
            scheme_labels = labels["schemes"][scheme]
            region = scheme_labels["site_label"]
            if region == "n/a":
                n_no_label += 1
                continue
            disp_cond = rec["displacement_conditions"].get("excluding_stimulated_shank", {})
            if disp_cond.get("status") != "computed" or disp_cond.get("normalised_displacement") is None:
                continue
            outcome = abs(disp_cond["normalised_displacement"])
            region_rows[region].append((rec["session_key"], rec["subject"], outcome))
            lobe_rows[scheme_labels["lobe"]].append((rec["session_key"], rec["subject"], outcome))

        def _n_subjects(rows):
            return len({s for _, s, _ in rows})

        qualifying = {label: rows for label, rows in region_rows.items() if _n_subjects(rows) >= MIN_SUBJECTS_PER_REGION_CELL}
        pooled_to_lobe = False
        cell_kind = "region"
        if not qualifying:
            qualifying = {label: rows for label, rows in lobe_rows.items()
                         if label != "unmapped_lobe" and _n_subjects(rows) >= MIN_SUBJECTS_PER_REGION_CELL}
            pooled_to_lobe = True
            cell_kind = "lobe"

        cell_sizes = {"region": {label: {"n_sessions": len(rows), "n_subjects": _n_subjects(rows)}
                                 for label, rows in region_rows.items()},
                      "lobe": {label: {"n_sessions": len(rows), "n_subjects": _n_subjects(rows)}
                              for label, rows in lobe_rows.items()}}

        if not qualifying:
            schemes_out[scheme] = {
                "status": "computed", "branch": "too_few_subjects_per_stimulation_region",
                "cell_sizes": cell_sizes, "n_sessions_no_label": n_no_label,
                "min_subjects_per_cell": MIN_SUBJECTS_PER_REGION_CELL,
            }
            continue

        all_rows = region_rows if not pooled_to_lobe else lobe_rows
        all_sessions = [(k, s, v, label) for label, rows in all_rows.items() for (k, s, v) in rows]
        subj_all = [s for _, s, _, _ in all_sessions]

        cells = {}
        p_values = []
        cell_order = []
        for label, rows in qualifying.items():
            member_sessions = {k for k, _, _ in rows}
            indicator = np.array([1.0 if k in member_sessions else 0.0 for k, _, _, _ in all_sessions])
            outcomes = np.array([v for _, _, v, _ in all_sessions])
            corr = subject_aggregated_correlation(indicator, outcomes, subj_all)
            cell_outcome = np.array([v for _, _, v in rows])
            cell_subject = [s for _, s, _ in rows]
            cell_mean = subject_clustered_mean_test(cell_outcome, cell_subject, alternative="two-sided")
            cells[label] = {"n_sessions": len(rows), "n_subjects": _n_subjects(rows),
                            "mean_abs_normalised_displacement_shank_excluded": cell_mean,
                            "correlation_of_membership_with_outcome": corr}
            if corr.get("status") == "computed":
                p_values.append(corr["p_value"])
                cell_order.append(label)

        if p_values:
            fdr = fdr_bh(np.array(p_values), alpha=ALPHA)
            for label, q, reject in zip(cell_order, fdr["q_values"].tolist(), fdr["reject"].tolist()):
                cells[label]["q_value"] = float(q)
                cells[label]["significant_after_fdr"] = bool(reject)
            any_significant = bool(fdr["n_reject"] > 0)
        else:
            any_significant = False

        branch = ("stimulation_site_region_is_associated_with_the_effect" if any_significant
                 else "no_regional_association_above_the_reported_bound")

        schemes_out[scheme] = {
            "status": "computed", "branch": branch, "cell_kind": cell_kind, "pooled_to_lobe": pooled_to_lobe,
            "cells": cells, "cell_sizes": cell_sizes, "n_sessions_no_label": n_no_label,
            "min_subjects_per_cell": MIN_SUBJECTS_PER_REGION_CELL,
        }

    return {"schemes": schemes_out, "coverage_confound_disclosure": (
        "Electrode placement in both corpora is chosen for clinical seizure localisation; no "
        "subject was randomised to a stimulation site. Every region or lobe cell above pools "
        "whatever subjects happened to be implanted and stimulated there. This block never "
        "yields a causal region claim regardless of which branch fires."
    )}


# ── Block C -- replication in the classifier-triggered corpus ──────────────

def run_block_c(open_loop_a: dict, open_loop_b: dict, closed_sessions: list[dict],
                closed_geometry: dict, closed_labels: dict) -> dict:
    block_a = run_block_a(closed_sessions, closed_geometry)
    block_b = run_block_b(closed_sessions, closed_labels)

    def _sign_agreement(open_corr_by_cond: dict, closed_corr_by_cond: dict) -> dict:
        out = {}
        for cond in CONDITION_NAMES:
            o, c = open_corr_by_cond.get(cond, {}), closed_corr_by_cond.get(cond, {})
            if o.get("status") == "computed" and c.get("status") == "computed":
                out[cond] = {"open_loop_r": o["r"], "closed_loop_r": c["r"],
                            "sign_agrees": (o["r"] < 0) == (c["r"] < 0)}
            else:
                out[cond] = {"status": "not_computable"}
        return out

    return {
        "causal": False,
        "causal_disclosure": (
            "This corpus's stimulation assignment is triggered by an online classifier reading "
            "the subject's own encoding-period state, not scheduled by the experimenter. Every "
            "number in this block carries causal:false and is never pooled with the open-loop "
            "corpus into a single headline."
        ),
        "block_a": block_a,
        "block_b": block_b,
        "sign_agreement_with_open_loop_displacement_relationship": _sign_agreement(
            open_loop_a["displacement_relationship"]["by_channel_condition"],
            block_a["displacement_relationship"]["by_channel_condition"]),
        "sign_agreement_with_open_loop_behavior_relationship": _sign_agreement(
            open_loop_a["behavior_relationship"]["by_channel_condition"],
            block_a["behavior_relationship"]["by_channel_condition"]),
    }


# ── Block D -- what the non-human site evidence adds, and what it cannot ───

def run_block_d(n_open_loop_subjects: int, n_open_loop_sessions: int) -> dict:
    macaque_pfc_microstimulation_path = RESULTS / "causal_macaque_pfc_microstimulation.json"
    non_human = {"status": "unavailable"}
    if macaque_pfc_microstimulation_path.exists():
        macaque_pfc_microstimulation = json.loads(macaque_pfc_microstimulation_path.read_text())
        per_session = macaque_pfc_microstimulation.get("per_session", {})
        animals = sorted({re.match(r"^[A-Za-z]+", k).group(0) for k in per_session if re.match(r"^[A-Za-z]+", k)})
        non_human = {"status": "computed", "n_sessions": len(per_session), "n_animals": len(animals)}

    return {
        "non_human_site_evidence": non_human,
        "human_site_evidence": {"corpus": "open_loop_ds005489", "n_subjects": n_open_loop_subjects,
                                "n_sessions": n_open_loop_sessions},
        "statements": [
            (
                "The non-human corpus stimulates inside a maintenance delay at single-unit "
                "resolution and can resolve the direction of the stimulation-evoked shift against "
                "the population's own geometry -- whether the shift moves along, across, or off the "
                "axis the memorandum occupies. This human corpus cannot do that: it stimulates "
                "during list encoding, not a maintenance delay, and it records field-potential power "
                "on macroelectrode contacts, not single-unit activity, so no comparable population "
                "geometry is available to resolve a direction against."
            ),
            (
                f"This human corpus resolves stimulation site across {n_open_loop_subjects} different "
                f"brains' worth of clinically placed electrodes in the open-loop corpus alone "
                f"({n_open_loop_sessions} sessions). The non-human corpus stimulates "
                f"{non_human.get('n_animals', 'a small fixed number of')} animals' own fixed implant "
                f"sites across all {non_human.get('n_sessions', 'its')} of its sessions and cannot ask "
                "a cross-brain site question at all."
            ),
            (
                "Holding both narrows one targeting hypothesis: a site-distance relationship found "
                "in this human corpus and a directionally resolved stimulation effect found in the "
                "non-human corpus would jointly support that placing a human electrode near the "
                "component-carrying tissue, during a period with a resolvable population direction, "
                "is what a targeting rule should ask for -- neither corpus alone can establish both "
                "halves of that claim."
            ),
            (
                "What would settle what the non-human result can only suggest: a human intracranial "
                "stimulation dataset with a genuine, isolated working-memory maintenance delay -- "
                "stimulation delivered after encoding and before probe, not overlapping either -- "
                "recorded with sufficient channel count to fit the same population-geometry direction "
                "test the non-human corpus supports. Neither human corpus analysed here has that "
                "delay period; both stimulate during encoding by design."
            ),
        ],
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", type=int, default=None,
                        help="limit each corpus to the first N admitted sessions, for a fast dev-time check")
    args = parser.parse_args()
    t0 = time.time()

    all_sessions = load_admitted_sessions()
    open_sessions = [s for s in all_sessions if s["corpus"] == "open_loop_ds005489"]
    closed_sessions = [s for s in all_sessions if s["corpus"] == "closed_loop_ds005557"]
    if args.smoke is not None:
        open_sessions = open_sessions[:args.smoke]
        closed_sessions = closed_sessions[:args.smoke]
    n_seen = len(open_sessions) + len(closed_sessions)

    open_geometry = _geometry_rows(open_sessions)
    closed_geometry = _geometry_rows(closed_sessions)

    open_labels = {s["session_key"]: run_checkpointed(
        f"labels__{s['corpus']}__{s['session_key']}",
        lambda s=s: site_labels_for_session(s["session_key"], s["corpus"], s["anode"], s["cathode"]))
        for s in open_sessions}
    closed_labels = {s["session_key"]: run_checkpointed(
        f"labels__{s['corpus']}__{s['session_key']}",
        lambda s=s: site_labels_for_session(s["session_key"], s["corpus"], s["anode"], s["cathode"]))
        for s in closed_sessions}

    block_a = run_block_a(open_sessions, open_geometry)
    block_b = run_block_b(open_sessions, open_labels)
    block_c = run_block_c(block_a, block_b, closed_sessions, closed_geometry, closed_labels)

    n_subjects_open = len({s["subject"] for s in open_sessions})
    block_d = run_block_d(n_subjects_open, len(open_sessions))

    geometry_exclusion_counts = defaultdict(int)
    for rows in (open_geometry, closed_geometry):
        for v in rows.values():
            if v.get("status") != "computed":
                geometry_exclusion_counts[v.get("reason", "unknown")] += 1
    behavior_exclusion_counts = defaultdict(int)
    for s in all_sessions:
        if s["behavior"]["status"] != "computed":
            behavior_exclusion_counts[s["behavior"]["reason"]] += 1
    label_exclusion_counts = defaultdict(int)
    for rows in (open_labels, closed_labels):
        for v in rows.values():
            if v.get("status") != "computed":
                label_exclusion_counts[v.get("reason", "unknown")] += 1

    output = {
        "version": "2026-08-22",
        "scope": (
            "Human intracranial free-recall stimulation, two corpora, joined session by session to "
            "an already-delivered component-displacement artifact and an already-delivered "
            "behavioural artifact -- neither is recomputed here. Session is the unit of analysis; "
            "subject is the clustering unit for every correlation and every pooled statistic. "
            "Block A and Block B run on the open-loop (experimenter-scheduled, causal) corpus; "
            "Block C repeats both in the classifier-triggered corpus, which is never pooled with "
            "the open-loop corpus into one headline."
        ),
        "zero_drop_accounting": {
            "n_sessions_seen_total": n_seen,
            "n_sessions_open_loop": len(open_sessions), "n_sessions_closed_loop": len(closed_sessions),
            "n_subjects_open_loop": n_subjects_open,
            "n_subjects_closed_loop": len({s["subject"] for s in closed_sessions}),
            "geometry_exclusions_by_reason": dict(geometry_exclusion_counts),
            "behavior_exclusions_by_reason": dict(behavior_exclusion_counts),
            "label_exclusions_by_reason": dict(label_exclusion_counts),
        },
        "block_a": block_a,
        "block_b": block_b,
        "block_c": block_c,
        "block_d": block_d,
    }

    output["wall_clock_s"] = time.time() - t0
    output["code_commit"] = git_commit(ROOT)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rendered = canonical_json(output)
    if "Infinity" in rendered or "NaN" in rendered:
        raise RuntimeError("non-finite token leaked into JSON output -- fix the offending field before writing")
    fd, tmp_name = tempfile.mkstemp(dir=str(OUTPUT_PATH.parent), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(rendered)
        os.replace(tmp_name, OUTPUT_PATH)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
    print(f"Wrote {OUTPUT_PATH} ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
