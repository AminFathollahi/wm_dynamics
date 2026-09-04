"""Guards on the two properties every claim in the human content-decodability
artifact rests on: that the picture-category grain keeps yielding five balanced
classes per session, and that every arm is one row per recording session.

The grain guard is split in two so it fails for the right reason. The derivation
itself is checked on constructed identifiers, so a change to how a category code
is read off a picture identifier fails here with no data on disk; the delivered
per-session class census is then checked, so a corpus whose pictures stop being
drawn evenly from five categories fails here too.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _extra in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from run_human_content_decodability import (  # noqa: E402
    EXPECTED_PICTURE_CATEGORIES,
    MIN_TRIALS_PER_CLASS,
    OUTPUT_PATH,
    PICTURE_CATEGORY_DIVISOR,
    STRUCTURE_OF_ANALYSIS,
    class_counts,
    picture_category_labels,
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not OUTPUT_PATH.exists():
        pytest.skip(f"{OUTPUT_PATH.name} has not been produced yet")
    return json.loads(OUTPUT_PATH.read_text())


def test_category_grain_collapses_picture_identity_to_five_classes():
    picture_ids = np.concatenate([
        np.arange(category * PICTURE_CATEGORY_DIVISOR, category * PICTURE_CATEGORY_DIVISOR + 26)
        for category in range(1, EXPECTED_PICTURE_CATEGORIES + 1)
    ])
    identity_counts = class_counts(picture_ids)
    category_counts = class_counts(picture_category_labels(picture_ids))

    assert len(category_counts) == EXPECTED_PICTURE_CATEGORIES
    assert set(category_counts.values()) == {26}
    assert max(identity_counts.values()) == 1, (
        "every picture is shown once, so identity grain must yield no class with repeats"
    )
    assert all(n < MIN_TRIALS_PER_CLASS for n in identity_counts.values())


def test_category_grain_is_not_a_relabelling_of_identity():
    picture_ids = np.array([101, 102, 201, 202, 305])
    assert list(picture_category_labels(picture_ids)) == [1, 1, 2, 2, 3]


def test_delivered_sessions_still_carry_five_balanced_categories(artifact):
    census = artifact["zero_drop_accounting"]["dandi_001187"]["per_session"]
    admitted = [row for row in census if row["admission"] == "admitted"]
    assert admitted, "no session was admitted, so the grain guard has nothing to check"

    for row in admitted:
        counts = row["picture_category_grain_class_counts"]
        assert len(counts) == EXPECTED_PICTURE_CATEGORIES, (
            f"{row['session']} yields {len(counts)} picture categories"
        )
        assert min(counts.values()) >= MIN_TRIALS_PER_CLASS, (
            f"{row['session']} has a category with fewer than {MIN_TRIALS_PER_CLASS} trials"
        )
        identity = row["picture_identity_grain_class_counts"]
        assert max(identity.values()) < MIN_TRIALS_PER_CLASS, (
            f"{row['session']} now repeats a picture often enough for identity grain to be usable, "
            "which changes what the category grain is a fallback from"
        )


def test_category_balance_stays_within_a_factor_of_two(artifact):
    census = artifact["zero_drop_accounting"]["dandi_001187"]["per_session"]
    for row in [r for r in census if r["admission"] == "admitted"]:
        counts = list(row["picture_category_grain_class_counts"].values())
        assert max(counts) <= 2 * min(counts), (
            f"{row['session']} category counts {counts} are no longer balanced"
        )


def test_every_arm_is_restricted_to_pooled_structure(artifact):
    for dataset, rows in artifact["zero_drop_accounting"]["delivered_arms"].items():
        assert rows["n_rows_used_at_pooled_structure"] <= rows["n_rows_delivered_all_structures"]
        assert rows["n_rows_dropped_as_per_structure_subsets"] >= 0, dataset

    for row in artifact["session_rows_dandi_001187"]:
        assert row["structure"] == STRUCTURE_OF_ANALYSIS, (
            f"{row['session']} entered the new arm at structure {row['structure']}"
        )

    for dataset, arm in artifact["primary_decodability"]["per_arm"].items():
        assert arm["structure"] == STRUCTURE_OF_ANALYSIS, dataset


def test_pooled_session_counts_match_one_row_per_recording(artifact):
    arms = artifact["primary_decodability"]["per_arm"]
    delivered = artifact["zero_drop_accounting"]["delivered_arms"]
    for dataset, rows in delivered.items():
        assert arms[dataset]["n_sessions"] == rows["n_rows_used_at_pooled_structure"], dataset
    assert arms["dandi_001187"]["n_sessions"] == len(artifact["session_rows_dandi_001187"])


def test_zero_drop_counts_reconcile(artifact):
    census = artifact["zero_drop_accounting"]["dandi_001187"]
    assert census["reconciliation_holds"], census["admission_counts"]
    assert sum(census["admission_counts"].values()) == census["n_canonical_sessions_seen"]
