from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from subject_independence import count_independent_groups, resolve_group  # noqa: E402


def test_collapsing_pair_counts_as_one_group():
    assert count_independent_groups(["boran", "dandi000574_units"]) == 1


def test_disjoint_corpora_stay_separate():
    assert count_independent_groups(["dandi000469", "pfc4", "watters_2026"]) == 3


def test_unregistered_identifier_raises():
    with pytest.raises(KeyError):
        resolve_group("not_a_real_corpus_identifier")


def test_ctg_six_cells_span_four_groups():
    cells = ["dandi000469", "boran", "dandi000574_units", "dandi000673", "dandi001187", "miller"]
    assert count_independent_groups(cells) == 4
