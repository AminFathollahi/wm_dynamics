#!/usr/bin/env python3
"""Feasibility gate: does ds005489 have >=~8 subjects with
>=2 DISTINCT stim SITES (different anode_label/cathode_label pairs across
their sessions)? A within-subject alignment-vs-effect test needs across-site
variation WITHIN a subject; >=2 sessions at the SAME site only adds
v*-estimation noise, not site variation.

Parses anode_label/cathode_label directly from every session's *_events.tsv
(the same fields run_ram_openloop_pipeline.build_session_features reads),
scanning ALL sessions with a stim word present -- not just the subset that
passes build_session_features's quality filters (MIN_WORDS, EDF-channel-name
match, etc.) -- since the feasibility question is about label diversity in
the raw BIDS metadata, independent of downstream analysis-quality exclusions.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_ram_within_subject_feasibility.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
DATA = dataset_path("ram_ds005489_openloop")
MIN_MULTISITE_SUBJECTS = 8


def _load_events(events_tsv: Path) -> list[dict]:
    with open(events_tsv) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main() -> None:
    ieeg_jsons = sorted(DATA.glob("sub-*/ses-*/ieeg/*_acq-bipolar_ieeg.json"))
    print(f"Found {len(ieeg_jsons)} candidate bipolar+stim session JSONs")

    subject_sites = defaultdict(set)
    subject_sessions = defaultdict(list)
    n_sessions_with_stim = 0

    for ieeg_json in ieeg_jsons:
        with open(ieeg_json) as f:
            meta = json.load(f)
        if not meta.get("ElectricalStimulation", False):
            continue
        stem = str(ieeg_json).replace("_ieeg.json", "")
        events_tsv = Path(stem.replace("_acq-bipolar", "") + "_events.tsv")
        if not events_tsv.exists():
            continue
        events = _load_events(events_tsv)
        words = [e for e in events if e.get("trial_type") == "WORD"]
        stim_word = next((w for w in words if w.get("stimulation") == "1"), None)
        if stim_word is None or stim_word.get("anode_label", "n/a") == "n/a":
            continue
        anode, cathode = stim_word["anode_label"], stim_word["cathode_label"]
        site = tuple(sorted([anode, cathode]))  # order-invariant site identity

        subj = ieeg_json.parts[-4]
        n_sessions_with_stim += 1
        subject_sites[subj].add(site)
        subject_sessions[subj].append({
            "session": str(ieeg_json.relative_to(DATA)), "anode": anode, "cathode": cathode,
        })

    multisite_subjects = {s: sites for s, sites in subject_sites.items() if len(sites) >= 2}
    n_multisite = len(multisite_subjects)
    n_total_subjects = len(subject_sites)

    print(f"\n{n_sessions_with_stim} sessions with a valid stim anode/cathode label "
          f"across {n_total_subjects} subjects")
    for subj in sorted(subject_sites):
        sites = subject_sites[subj]
        tag = "MULTI-SITE" if len(sites) >= 2 else "single-site"
        print(f"  {subj}: {len(subject_sessions[subj])} stim session(s), "
              f"{len(sites)} distinct site(s) [{tag}] -- {sorted(sites)}")

    print(f"\nn_multisite_subjects (>=2 distinct anode/cathode sites) = {n_multisite}")
    print(f"Gate threshold: >= {MIN_MULTISITE_SUBJECTS}")

    if n_multisite < MIN_MULTISITE_SUBJECTS:
        out = {
            "status": "underpowered",
            "n_multisite_subjects": n_multisite,
            "n_total_subjects_with_stim": n_total_subjects,
            "n_sessions_with_stim": n_sessions_with_stim,
            "threshold": MIN_MULTISITE_SUBJECTS,
            "reason": ("RAM is ~one-site-per-subject; a within-subject graded test is not "
                      "constructible"),
            "multisite_subjects": {s: [list(x) for x in sites] for s, sites in multisite_subjects.items()},
            "per_subject_site_count": {s: len(sites) for s, sites in subject_sites.items()},
        }
        with open(RESULTS / "causal_ram_within_subject.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nGATE FAILED: n_multisite_subjects={n_multisite} < {MIN_MULTISITE_SUBJECTS} -- "
              "wrote underpowered status, stopping (keep the pooled null as the honest bound).")
    else:
        print(f"\nGATE PASSED: n_multisite_subjects={n_multisite} >= {MIN_MULTISITE_SUBJECTS} -- "
              "proceed to the within-subject refit (not yet implemented in this script).")
        # feasibility-only script; the refit (if the gate passes) is a separate step per spec.
        out = {
            "status": "feasible_gate_only",
            "n_multisite_subjects": n_multisite,
            "n_total_subjects_with_stim": n_total_subjects,
            "multisite_subjects": {s: [list(x) for x in sites] for s, sites in multisite_subjects.items()},
        }
        with open(RESULTS / "causal_ram_within_subject.json", "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
