import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _subdir in ("src", "scripts"):
    sys.path.insert(0, str(ROOT / _subdir))

from statistics import stable_seed  # noqa: E402

from test_producer_code_identity import PRODUCERS  # noqa: E402


def _unseeded_calls(tree):
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "hash":
            found.append((node.lineno, "hash()"))
        if isinstance(func, ast.Attribute) and func.attr == "default_rng" and not node.args and not node.keywords:
            found.append((node.lineno, "default_rng() without a seed"))
    return found


@pytest.mark.parametrize("producer", PRODUCERS)
def test_producer_has_no_process_salted_or_unseeded_randomness(producer):
    tree = ast.parse((ROOT / "scripts" / f"{producer}.py").read_text())
    assert _unseeded_calls(tree) == []


def test_ondemand_streak_generator_depends_only_on_the_cohort_key():
    from run_ondemand_streak_diagnostic import _cohort_rng

    first = _cohort_rng("boran_sub-01").integers(0, 2**31 - 1, size=5)
    again = _cohort_rng("boran_sub-01").integers(0, 2**31 - 1, size=5)
    expected = np.random.default_rng(stable_seed("boran_sub-01")).integers(0, 2**31 - 1, size=5)
    np.testing.assert_array_equal(first, again)
    np.testing.assert_array_equal(first, expected)
    assert not np.array_equal(first, _cohort_rng("miller_al").integers(0, 2**31 - 1, size=5))
