#!/usr/bin/env python3
"""The zero-drop narrative map: one entry per file in results/, so that no
artifact can silently disappear from the record. Reduced in scope from an
earlier spec that also allocated each result to a main-text/supplement
position by hand -- that allocation will change once the full battery has
run, so what is kept here is the machine-checkable half of the zero-drop
rule: every file gets a home, a claim, and an evidence-gate tag, and the
test suite fails loudly if a later round adds a result without one.

For the 149 JSON analysis artifacts, `claim` and `verdict` are extracted
directly from each artifact's own fields (its `trigger`/`scope` text and its
`predeclared_decision.verdict`/top-level `verdict`/`status`, in that
preference order) wherever present, falling back to a generic
`analysis_id`-derived claim only when an artifact carries none of those
fields. `evidence_gate` is assigned by a keyword heuristic over the
filename and analysis_id against VALID_GATES (src/provenance.py) -- a rough
first pass, not a claim of curatorial judgment on each of 149 files, and
named as such in the artifact's own `home_assignment_method` field.

Non-JSON files in results/ (raw .npz caches from the notebook-era pipeline,
.pdf figures, .md documents, .log/.lock files) are not independent claims,
so each gets a minimal, explicit "not an independent claim" entry rather
than either a fabricated one or a silent exclusion -- the file still counts
toward the "every file appears exactly once" invariant the test enforces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance import VALID_GATES, canonical_json, git_commit, sha256_file  # noqa: E402

RESULTS = ROOT / "results"
LEDGER = ROOT / "provenance" / "evidence_ledger.json"
HOME_VALUES = ("main_text", "supplementary", "methods_only", "superseded_recorded")
DISPOSITION_VALUES = ("evidence", "plumbing", "superseded", "excluded")

GATE_KEYWORDS = {
    "G0": ("identity", "census", "registry", "eligibility"),
    "G3": ("stim", "perturbation", "causal", "opto", "closed_loop", "openloop", "macaque_pfc_microstimulation", "tes1", "amplification"),
    "G4": ("control", "lqr", "digital_twin", "rl_policy", "targeting", "controllability"),
    "G2": (
        "drift", "confinement", "lambda", "identifiability", "dynamics", "rotation", "switching",
        "displacement", "msd", "attractor", "dmd", "divergence", "tangling", "eigenspectr",
        "recurrence", "homology", "microcircuit", "graph",
    ),
    "G1": (
        "decoding", "ctg", "rsa", "content", "geometry", "dimensionality", "representation",
        "crossnobis", "axis", "capacity", "pr_", "participation_ratio",
    ),
}

METHODS_ONLY_KEYWORDS = ("calibration", "estimator_limits", "positive_control", "robustness", "sensitivity", "gate_", "simulation", "synthetic")
SECTION_3_TO_8_ANALYSIS_IDS = {
    "attractor_recovery_control", "intrinsic_dimensionality", "latent_displacement_scaling",
    "functional_microcircuit_graph", "structure_identifiability_model",
}


def gate_for(name: str) -> str:
    lowered = name.lower()
    for gate, keywords in GATE_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return gate
    return "G1"  # most artifacts in this project characterize population representational content


def home_for(analysis_id: str, filename: str, has_superseded_by: bool) -> str:
    if has_superseded_by:
        return "superseded_recorded"
    lowered = f"{analysis_id} {filename}".lower()
    if analysis_id in SECTION_3_TO_8_ANALYSIS_IDS:
        return "main_text"
    if any(kw in lowered for kw in METHODS_ONLY_KEYWORDS):
        return "methods_only"
    return "supplementary"


def find_first(d: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def find_verdict(d: dict) -> str | None:
    for path in (
        ("predeclared_decision", "verdict"),
        ("verdict",),
        ("status",),
        ("predeclared_interpretation", "verdict"),
        ("gate_decision",),
    ):
        node = d
        ok = True
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                ok = False
                break
        if ok and isinstance(node, (str, bool, type(None))):
            return str(node)
    return None


def disposition_for_status(status: str) -> tuple[str, str]:
    """Map a ledger status to its narrative disposition."""
    if status == "pipeline_provenance":
        return "plumbing", "records pipeline operation rather than an independent finding"
    if status in {"superseded", "invalidated_do_not_interpret"}:
        return "superseded", f"evidence ledger status is {status}"
    if status == "skipped":
        return "excluded", "analysis was skipped under its recorded admission rule"
    return "evidence", f"evidence ledger status is {status}"


def entry_for_json(path: Path, ledger_by_path: dict[str, dict]) -> dict:
    relative = str(path.relative_to(ROOT))
    if path.name.endswith(".shakedown.json"):
        return {
            "artifact": relative, "home": "methods_only",
            "claim": "reduced-settings execution check; not a scientific result",
            "evidence_gate": "G0", "verdict": "not_applicable",
            "superseded_by": None, "retired_by": None,
            "home_assignment_method": "reduced_settings_exclusion",
            "disposition": "excluded",
            "disposition_reason": "reduced settings are not comparable to the production analysis",
        }
    ledger_row = ledger_by_path.get(relative)
    if ledger_row is None:
        raise RuntimeError(f"JSON artifact has no evidence-ledger row: {relative}")
    disposition, disposition_reason = disposition_for_status(ledger_row["status"])
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "artifact": relative, "home": "supplementary",
            "claim": f"file did not parse as JSON ({exc}); not machine-readable",
            "evidence_gate": "G1", "verdict": "not_applicable",
            "superseded_by": None, "retired_by": None, "home_assignment_method": "parse_failure_fallback",
            "disposition": disposition, "disposition_reason": disposition_reason,
        }
    if not isinstance(data, dict):
        return {
            "artifact": relative, "home": "supplementary",
            "claim": "JSON root is not an object; no structured claim fields to extract",
            "evidence_gate": "G1", "verdict": "not_applicable",
            "superseded_by": None, "retired_by": None, "home_assignment_method": "non_object_root_fallback",
            "disposition": disposition, "disposition_reason": disposition_reason,
        }

    analysis_id = data.get("analysis_id", path.stem)
    claim = find_first(data, ("trigger", "scope", "note"))
    if claim is None:
        claim = f"see {analysis_id} for its computed results (no trigger/scope field to quote directly)"
    claim = str(claim)[:400]

    verdict = find_verdict(data)
    if verdict is None:
        verdict = "not_applicable"

    superseded_by = data.get("superseded_by")
    retired_by = data.get("retired_by") or data.get("superseded_by")
    home = home_for(analysis_id, path.name, superseded_by is not None)

    return {
        "artifact": relative, "home": home, "claim": claim,
        "evidence_gate": gate_for(f"{analysis_id} {path.name}"), "verdict": verdict,
        "superseded_by": superseded_by, "retired_by": retired_by if home == "superseded_recorded" else None,
        "home_assignment_method": "extracted_from_artifact_fields_plus_keyword_heuristic",
        "disposition": disposition, "disposition_reason": disposition_reason,
    }


def entry_for_non_json(path: Path) -> dict:
    relative = str(path.relative_to(ROOT))
    suffix = path.suffix.lstrip(".")
    kind = {
        "npz": "raw numpy data cache from the notebook-era pipeline",
        "pdf": "rendered figure",
        "md": "prose document, not an independent quantitative claim",
        "log": "run log",
        "lock": "file lock artifact",
        "logs": "run log directory",
    }.get(suffix, f"{suffix} file")
    disposition = "evidence" if suffix == "pdf" else "plumbing"
    disposition_reason = (
        "rendered presentation of a quantitative result"
        if disposition == "evidence"
        else f"{kind} supporting reproducibility rather than an independent finding"
    )
    return {
        "artifact": relative, "home": "supplementary",
        "claim": f"{kind}; not an independent claim -- see its generating script/artifact for the claim it supports",
        "evidence_gate": gate_for(path.name), "verdict": "not_applicable",
        "superseded_by": None, "retired_by": None, "home_assignment_method": "non_json_generic_entry",
        "disposition": disposition, "disposition_reason": disposition_reason,
    }


def build_supplementary_index(entries: list[dict]) -> str:
    lines = ["# Supplementary Index", "", "Generated by scripts/build_narrative_map.py from results/narrative_map.json. Do not hand-edit.", ""]
    for home in HOME_VALUES:
        rows = [e for e in entries if e["home"] == home]
        lines.append(f"## {home} ({len(rows)})")
        lines.append("")
        for row in sorted(rows, key=lambda e: e["artifact"]):
            gate = row["evidence_gate"]
            verdict = row["verdict"]
            disposition = row["disposition"]
            lines.append(f"- `{row['artifact']}` [{gate}; {disposition}] verdict={verdict}: {row['claim']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ledger_rows = json.loads(LEDGER.read_text())
    ledger_by_path = {row["artifact_path"]: row for row in ledger_rows}
    files = sorted(p for p in RESULTS.iterdir() if p.is_file())
    entries = []
    for path in files:
        if path.suffix == ".json":
            entries.append(entry_for_json(path, ledger_by_path))
        else:
            entries.append(entry_for_non_json(path))
    if (RESULTS / "narrative_map.json") not in files:
        # self-referential: this script's own output is a file in results/ too, and must be
        # covered from its very first run, not only once a second run sees it already on disk.
        entries.append({
            "artifact": "results/narrative_map.json", "home": "methods_only",
            "claim": "this map itself: one entry per file in results/, generated by scripts/build_narrative_map.py",
            "evidence_gate": "G0", "verdict": "not_applicable",
            "superseded_by": None, "retired_by": None, "home_assignment_method": "self_reference",
            "disposition": "evidence",
            "disposition_reason": "machine-readable index of the evidence record",
        })

    home_counts = {home: sum(1 for e in entries if e["home"] == home) for home in HOME_VALUES}
    disposition_counts = {
        value: sum(1 for entry in entries if entry["disposition"] == value)
        for value in DISPOSITION_VALUES
    }
    output = {
        "schema_version": "1.1.0",
        "analysis_id": "narrative_map",
        "code_commit": git_commit(ROOT),
        "source_hash": sha256_file(Path(__file__)),
        "scope": (
            "Machine-checkable zero-drop map only: every file in results/ gets a home, claim, "
            "evidence_gate, and verbatim verdict. The main-text/supplement editorial allocation "
            "this map's `home` field approximates by keyword heuristic is deferred to the actual "
            "manuscript-writing pass; this is not that pass."
        ),
        "n_files_covered": len(entries),
        "home_counts": home_counts,
        "disposition_counts": disposition_counts,
        "entries": entries,
    }
    (RESULTS / "narrative_map.json").write_text(canonical_json(output))
    (ROOT / "SUPPLEMENTARY_INDEX.md").write_text(build_supplementary_index(entries))
    print(json.dumps({
        "n_files_covered": len(entries),
        "home_counts": home_counts,
        "disposition_counts": disposition_counts,
    }, indent=2))


if __name__ == "__main__":
    main()
