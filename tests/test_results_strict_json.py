import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).parent.parent / "results"


def _reject_nonfinite(token):
    raise ValueError(f"non-finite token {token!r} is not valid strict JSON")


@pytest.mark.parametrize("path", sorted(RESULTS.glob("*.json")), ids=lambda p: p.name)
def test_artifact_parses_as_strict_json(path):
    with open(path) as f:
        json.load(f, parse_constant=_reject_nonfinite)
