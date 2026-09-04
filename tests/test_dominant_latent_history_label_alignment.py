"""Alignment checks for the trial-history relabelling.

An off-by-one in either direction, or a shift applied after an outcome
restriction rather than before it, would silently produce a history effect
that is not there. These assert the alignment directly rather than testing a
downstream summary of it.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_dominant_latent_identity_and_behaviour_breadth import (  # noqa: E402
    circular_shift_labels, history_labels,
)


def test_each_trial_takes_the_item_from_exactly_lag_trials_earlier():
    items = np.array([10, 11, 12, 13, 14, 15])
    for lag in (1, 2, 3):
        shifted, defined = history_labels(items, lag)
        for i in range(lag, len(items)):
            assert shifted[i] == items[i - lag], f"lag {lag}, trial {i}"
        assert defined[lag:].all()


def test_the_first_lag_trials_have_no_predecessor_and_are_masked_out():
    items = np.array([10, 11, 12, 13, 14, 15])
    for lag in (1, 2, 5):
        _shifted, defined = history_labels(items, lag)
        assert not defined[:lag].any()
        assert int(defined.sum()) == len(items) - lag


def test_the_shift_never_reaches_forward_in_time():
    """A sign error would label a trial with a LATER item, which is not a
    history variable at all; nothing downstream would flag it."""
    items = np.arange(20) * 3
    shifted, defined = history_labels(items, 1)
    assert (shifted[defined] < items[defined]).all()


def test_shifting_after_an_outcome_restriction_would_skip_over_errors():
    """The predecessor is the trial that physically preceded, right or wrong.
    Shifting the already-restricted sequence instead gives the previous
    CORRECT trial's item, a different quantity; this pins the difference so a
    future reordering of the two operations fails here."""
    items = np.array([0, 1, 2, 3, 4, 5])
    correct = np.array([True, False, False, True, True, True])

    shift_then_restrict, defined = history_labels(items, 1)
    keep = correct & defined
    assert list(shift_then_restrict[keep]) == [2, 3, 4]

    restrict_then_shift, _ = history_labels(items[correct], 1)
    assert list(restrict_then_shift[1:]) == [0, 3, 4]
    assert list(shift_then_restrict[keep]) != list(restrict_then_shift[1:])


def test_a_zero_or_negative_lag_is_rejected_rather_than_returning_the_current_item():
    with pytest.raises(ValueError):
        history_labels(np.arange(5), 0)


def test_the_circular_shift_null_preserves_the_label_multiset_and_the_sequence_order():
    items = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    for offset in (2, 5, 7):
        shifted = circular_shift_labels(items, offset)
        assert sorted(shifted.tolist()) == sorted(items.tolist())
        assert len(shifted) == len(items)
        # Rolling is a rotation, so every adjacent pair of the original survives
        # except the one broken at the wrap point.
        original_pairs = list(zip(items[:-1], items[1:]))
        shifted_pairs = list(zip(shifted[:-1], shifted[1:]))
        assert len(set(original_pairs) - set(shifted_pairs)) <= 1


def test_the_circular_shift_null_stays_clear_of_every_tested_history_lag():
    """The null's offset floor exists so no replicate accidentally reproduces
    the current item or one of the history lags being tested."""
    from run_dominant_latent_identity_and_behaviour_breadth import (
        HISTORY_LAGS, MIN_CIRCULAR_SHIFT_OFFSET,
    )

    assert MIN_CIRCULAR_SHIFT_OFFSET > max(HISTORY_LAGS)
    assert MIN_CIRCULAR_SHIFT_OFFSET > 0
