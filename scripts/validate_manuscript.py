#!/usr/bin/env python3
"""Fail fast on manuscript/README/evidence-ledger regressions: stale test counts, forbidden
result wording, missing corpus-identity statements, an absent archive boundary, claims of
achieved causal control above that boundary, and citations that are not reconciled with the
evidence ledger in both directions."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "PAPER_REPORT.tex"
README = ROOT / "README.md"
LEDGER = ROOT / "provenance" / "evidence_ledger.json"

CURRENT_TEST_COUNT = 550
STALE_TEST_COUNTS = (341, 402, 408, 417, 446, 454, 483, 492, 500, 515, 535, 540)
FORBIDDEN_RESULTS_WORDS = re.compile(
    r"\btrends?\b|\btrending\b|\bmarginal(?:ly)?\b|\bnearly significant\b",
    re.IGNORECASE,
)


def validate_text(paper: str, readme: str, ledger: list[dict]) -> list[str]:
    errors: list[str] = []
    combined = paper + "\n" + readme

    if re.search(r"\b90[- ]recording", combined, re.IGNORECASE):
        errors.append("stale 90-recording pooled count")
    if re.search(r"\bunstable direction\b", combined, re.IGNORECASE):
        errors.append("unqualified 'unstable direction' label")
    if FORBIDDEN_RESULTS_WORDS.search(combined):
        errors.append("forbidden trend/marginal/nearly-significant wording")

    for count in STALE_TEST_COUNTS:
        patterns = (
            rf"\b{count}\s+tests?\b",
            rf"\btests?\s*(?:collect(?:s|ed)?|passed|:)?.{{0,16}}\b{count}\b",
            rf"\b{count}/{count}\b",
        )
        if any(re.search(pattern, combined, re.IGNORECASE) for pattern in patterns):
            errors.append(f"stale test count {count}")
    if not re.search(rf"\b{CURRENT_TEST_COUNT}\s+tests?\b", combined, re.IGNORECASE):
        errors.append(f"current test count {CURRENT_TEST_COUNT} absent")

    identity_window = re.compile(
        r"(?is)(?:001187.{0,500}000673|000673.{0,500}001187)"
    )
    for match in identity_window.finditer(paper):
        if re.search(r"different cohort", match.group(0), re.IGNORECASE):
            errors.append("001187/000673 described as different cohorts")
            break
    if "001187 and 000673 are NOT independent" not in paper:
        errors.append("canonical 001187/000673 identity statement absent")

    if "\\citep{barbosa2021}" not in paper or "\\citep{wolff2017}" not in paper:
        errors.append("both Wolff and Barbosa readings must be cited")
    if "supports neither reading\nas settled fact" not in paper:
        errors.append("Wolff shortcut is not explicitly non-adjudicating")

    archive_marker = "\\part{Retained exploratory results archive}"
    if archive_marker not in paper:
        errors.append("exploratory archive boundary absent")
        current_claim_text = paper
    else:
        current_claim_text = paper.split(archive_marker, 1)[0]
    positive_control_claims = (
        "we demonstrate control",
        "achieved control",
        "can be steered",
        "govern real stimulation",
        "validated causal target",
    )
    for phrase in positive_control_claims:
        if phrase in current_claim_text.lower():
            errors.append(f"control claim above passed gate: {phrase!r}")

    ledger_by_name = {Path(row["artifact_path"]).name: row for row in ledger}
    explicit_results = {
        name.replace("\\_", "_")
        for name in re.findall(r"results/([A-Za-z0-9_.\\-]+\.json)", paper)
    }
    for name in sorted(explicit_results):
        if name not in ledger_by_name:
            errors.append(f"manuscript result ID absent from evidence ledger: {name}")

    # "pipeline_provenance" rows record how the pipeline was run (environment, corpus staging,
    # fetch planning, loader reuse, feasibility screening, a recording-site inventory) rather than
    # what was found, so they are deliberately excluded from this set and never require a
    # manuscript citation.
    current_statuses = {
        "current_confirmatory", "current_exploratory",
        "replication_or_informative_null", "nonidentified",
    }
    for row in ledger:
        if row["status"] not in current_statuses:
            continue
        name = Path(row["artifact_path"]).name
        if name not in paper and name.replace("_", "\\_") not in paper:
            errors.append(f"current ledger artifact has no manuscript citation: {name}")

    # These three anchored the project's earlier confinement-rate/diffusion spine. That spine is
    # retired: build_evidence_ledger.py's RETIRED_SPINE_STEMS block deliberately reclassifies it (and
    # everything built on it) to "superseded" now that the permutation-based cross-unit-state result
    # is the current confirmatory result. This check is updated to match that deliberate
    # reclassification rather than the pre-restructuring statuses it used to require.
    required_current = {
        "human_drift_spine_000469.json": "superseded",
        "human_drift_behavior_000469.json": "superseded",
        "drift_control_payload_000469.json": "superseded",
    }
    for name, status in required_current.items():
        row = ledger_by_name.get(name)
        if row is None:
            errors.append(f"required current ledger row missing: {name}")
        elif row["status"] != status:
            errors.append(f"{name} has status {row['status']!r}, expected {status!r}")

    return errors


def main() -> None:
    ledger = json.loads(LEDGER.read_text())
    errors = validate_text(PAPER.read_text(), README.read_text(), ledger)
    if errors:
        raise SystemExit("Manuscript validation failed:\n- " + "\n- ".join(errors))
    print(
        f"Manuscript validation passed: {CURRENT_TEST_COUNT} tests declared, "
        f"{len(ledger)} ledger rows reconciled."
    )


if __name__ == "__main__":
    main()
