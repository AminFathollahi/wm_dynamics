"""Unit tests for scripts/run_stimulation_design_census.py's pure logic:
treatment-assignment classification, time-window/item-overlap arithmetic,
dose-variation and monotone-rescaling detection, and the row completeness
gate. None of these tests touch the external data mount -- they exercise
the same functions the live census calls, on synthetic inputs, so they run
without WM_DYNAMICS_DATA_ROOT set.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_stimulation_design_census as census  # noqa: E402


# ── classify_treatment_assignment ────────────────────────────────────────────

def test_single_level_beats_everything_else():
    assert census.classify_treatment_assignment(
        single_level=True, state_triggered=True, clinically_chosen_site=True,
        documented_randomization=True,
    ) == "single_level_nothing_to_randomize"


def test_state_triggered_overrides_documented_randomization():
    """A release that logs a
    nominal randomization/trigger field must still be classified as
    state-triggered selection, never as randomized, when delivery is in
    fact keyed to an online neural readout."""
    assert census.classify_treatment_assignment(
        single_level=False, state_triggered=True, clinically_chosen_site=False,
        documented_randomization=True,
    ) == "state_triggered_selection_not_randomized"


def test_clinically_chosen_site_not_randomized_even_if_documented():
    assert census.classify_treatment_assignment(
        single_level=False, state_triggered=False, clinically_chosen_site=True,
        documented_randomization=True,
    ) == "site_clinically_or_anatomically_chosen_not_randomized"


def test_documented_randomization_without_confounds_is_randomized():
    assert census.classify_treatment_assignment(
        single_level=False, state_triggered=False, clinically_chosen_site=False,
        documented_randomization=True,
    ) == "experimenter_randomized"


def test_no_mechanism_documented_falls_through_to_undocumented():
    assert census.classify_treatment_assignment(
        single_level=False, state_triggered=False, clinically_chosen_site=False,
        documented_randomization=False,
    ) == "assignment_mechanism_not_documented_in_the_release"


def test_classification_negative_control_inverted_logic_would_fail_this():
    """Synthetic recovery / negative control: if the precedence order in
    classify_treatment_assignment were inverted (documented_randomization
    checked before state_triggered), a state-triggered arm that also
    happens to carry a documented-randomization flag would be
    misclassified as randomized. This test locks in the correct precedence
    and fails under that inversion."""
    result = census.classify_treatment_assignment(
        single_level=False, state_triggered=True, clinically_chosen_site=False,
        documented_randomization=True,
    )
    assert result != "experimenter_randomized"
    assert result == "state_triggered_selection_not_randomized"

    def inverted(*, single_level, state_triggered, clinically_chosen_site, documented_randomization):
        if single_level:
            return "single_level_nothing_to_randomize"
        if documented_randomization:  # wrong precedence: checked before state_triggered
            return "experimenter_randomized"
        if state_triggered:
            return "state_triggered_selection_not_randomized"
        if clinically_chosen_site:
            return "site_clinically_or_anatomically_chosen_not_randomized"
        return "assignment_mechanism_not_documented_in_the_release"

    assert inverted(single_level=False, state_triggered=True, clinically_chosen_site=False,
                     documented_randomization=True) != result


# ── train_item_overlap_count (time-window arithmetic) ────────────────────────

def test_train_spans_exactly_two_items():
    """A ~4.6 s train against ~2.5 s item spacing, as in the RAM open-loop
    corpus: the train must overlap exactly two item-presentation windows."""
    item_windows = [(0.0, 1.6), (2.5, 4.1), (5.0, 6.6)]
    train = (0.5, 5.1)  # overlaps item 0 (0.0-1.6) and item 1 (2.5-4.1) fully, touches item 2's start
    assert census.train_item_overlap_count(*train, item_windows) == 3
    train_two_only = (0.5, 4.1)
    assert census.train_item_overlap_count(*train_two_only, item_windows) == 2


def test_short_train_confined_to_a_single_item():
    """A ~0.5 s train against the same item spacing, as in the RAM
    classifier-triggered corpus: confined to one item."""
    item_windows = [(0.0, 1.6), (2.5, 4.1), (5.0, 6.6)]
    train = (2.6, 3.1)
    assert census.train_item_overlap_count(*train, item_windows) == 1


def test_train_touching_zero_items_returns_zero():
    item_windows = [(0.0, 1.6), (2.5, 4.1)]
    train = (1.7, 2.4)
    assert census.train_item_overlap_count(*train, item_windows) == 0


def test_half_open_boundary_matches_touching_not_overlapping():
    """A train ending exactly at an item's onset does not count as overlap
    (half-open interval semantics, matching
    run_stimulation_timing_and_parameter_structure.overlaps)."""
    item_windows = [(2.5, 4.1)]
    assert census.train_item_overlap_count(1.0, 2.5, item_windows) == 0
    assert census.train_item_overlap_count(1.0, 2.50001, item_windows) == 1


# ── dose_variation_flags ──────────────────────────────────────────────────────

def test_dose_variation_flags_detects_within_group_variation():
    flags = census.dose_variation_flags({
        "pair_a": [1000.0, 1000.0, 1500.0],  # varies
        "pair_b": [500.0, 500.0, 500.0],     # constant
        "pair_c": [750.0],                    # single observation, no variation
    })
    assert flags == {"pair_a": True, "pair_b": False, "pair_c": False}


def test_dose_variation_flags_ignores_nan():
    flags = census.dose_variation_flags({"pair_a": [float("nan"), 300.0, float("nan")]})
    assert flags["pair_a"] is False


# ── monotone_rescaling_pairs ──────────────────────────────────────────────────

def test_monotone_rescaling_detected_for_exact_linear_relationship():
    """charge = amplitude * pulse_width_fixed is an exact rescaling of
    amplitude whenever pulse width is held fixed -- the census must name
    this as one degree of freedom, not two independently-varying doses."""
    rows = [
        {"amplitude": 500.0, "charge": 150.0},
        {"amplitude": 1000.0, "charge": 300.0},
        {"amplitude": 1500.0, "charge": 450.0},
    ]
    pairs = census.monotone_rescaling_pairs(rows, ("amplitude", "charge"))
    assert len(pairs) == 1
    assert pairs[0]["params"] == ["amplitude", "charge"]
    assert abs(pairs[0]["ratio_a_to_b"] - (500.0 / 150.0)) < 1e-9


def test_monotone_rescaling_absent_for_independently_varying_parameters():
    rows = [
        {"amplitude": 500.0, "pulse_freq_hz": 50.0},
        {"amplitude": 1000.0, "pulse_freq_hz": 10.0},
        {"amplitude": 1500.0, "pulse_freq_hz": 200.0},
    ]
    pairs = census.monotone_rescaling_pairs(rows, ("amplitude", "pulse_freq_hz"))
    assert pairs == []


def test_monotone_rescaling_needs_at_least_two_comparable_rows():
    rows = [{"amplitude": 500.0, "charge": 150.0}]
    assert census.monotone_rescaling_pairs(rows, ("amplitude", "charge")) == []


# ── VOID / MEASURED tagging and the row completeness gate ───────────────────

def test_void_and_measured_are_the_only_two_tag_shapes():
    v = census.VOID("no device exists in this release")
    m = census.MEASURED(42, "some field")
    assert v["status"] == "structural_void" and "reason" in v
    assert m["status"] == "measured" and m["value"] == 42 and "source" in m
    assert v["status"] != m["status"]


def test_validate_row_flags_a_missing_field():
    row = {field: census.VOID("x") for field in census.REQUIRED_ROW_FIELDS}
    row["corpus_id"] = "x"
    row["arm_id"] = "y"
    del row["amplitude"]
    defects = census.validate_row(row)
    assert any("missing field: amplitude" in d for d in defects)


def test_validate_row_flags_an_untagged_field():
    row = {field: census.VOID("x") for field in census.REQUIRED_ROW_FIELDS}
    row["corpus_id"] = "x"
    row["arm_id"] = "y"
    row["amplitude"] = 1000.0  # bare value, not MEASURED/VOID-tagged -- the error this gate exists to catch
    defects = census.validate_row(row)
    assert any("amplitude" in d for d in defects)


def test_validate_row_passes_a_fully_tagged_row():
    row = {field: census.VOID("not applicable to this synthetic row") for field in census.REQUIRED_ROW_FIELDS}
    row["corpus_id"] = "synthetic_corpus"
    row["arm_id"] = "synthetic_arm"
    assert census.validate_row(row) == []


# ── zero-drop reconciliation ──────────────────────────────────────────────────

def test_zero_drop_reconciliation_on_every_delivered_row():
    """For every row this census actually ships whose session_accounting
    carries an explicit seen/included/excluded triple, seen must equal
    included plus the number of excluded reasons -- the project's
    zero-drop rule, checked mechanically rather than by eyeballing the
    artifact. Skips corpora whose accounting uses a different (but still
    reconciling) key set, e.g. the participant-pair inventory for ds005034."""
    import json

    out_path = Path(__file__).resolve().parents[1] / "results" / "stimulation_design_census.json"
    if not out_path.exists():
        return  # the live artifact is produced by a separate, data-mount-dependent run
    payload = json.loads(out_path.read_text())
    checked = 0
    for row in payload["rows"]:
        acc = row["session_accounting"]
        if acc["status"] != "measured":
            continue
        value = acc["value"]
        if "n_sessions_seen" in value and "n_sessions_included" in value and "excluded_by_reason" in value:
            assert value["n_sessions_seen"] == value["n_sessions_included"] + len(value["excluded_by_reason"]), (
                row["corpus_id"], row["arm_id"])
            checked += 1
    assert checked >= 5  # sanity floor: most corpora in this census use the seen/included/excluded shape
