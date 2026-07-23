import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from run_targeting_benchmark import _near_tie_candidates, NEAR_TIE_REL_TOL


class TestNearTieCandidates:
    def test_returns_only_argmax_when_others_far_below_tolerance(self):
        scores = np.array([0.1, 0.2, 0.95, 0.3])
        assert _near_tie_candidates(scores) == [2]

    def test_includes_all_near_tie_donors_in_descending_order(self):
        scores = np.array([0.5, 0.91, 0.95, 0.2, 0.93])
        cands = _near_tie_candidates(scores)
        assert cands == [2, 4, 1]
        assert all(scores[cands[i]] >= scores[cands[i + 1]] for i in range(len(cands) - 1))

    def test_tolerance_is_relative_to_max(self):
        scores = np.array([1.0, NEAR_TIE_REL_TOL, NEAR_TIE_REL_TOL - 0.01])
        cands = _near_tie_candidates(scores)
        assert 0 in cands and 1 in cands and 2 not in cands

    def test_single_donor(self):
        assert _near_tie_candidates(np.array([0.42])) == [0]
