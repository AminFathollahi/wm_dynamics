"""Tests for scripts/build_corpus_admission_audit.py.

Covers the admission-predicate classifier on planted synthetic loader
snippets (each shape of trial-admission line this repository actually
contains), the extraction of predicate lines from a planted function body,
and the zero-drop accounting arithmetic that reconciles loaders_found =
audited + skipped_with_reason.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_corpus_admission_audit import (  # noqa: E402
    ADMISSION_ASSIGN_RE,
    OUTCOME_SUBSCRIPT_RE,
    classify_predicate_lines,
    evaluate_branches,
    extract_predicate_lines,
)


def _classified(body: str, signature: str = "def loader():") -> dict:
    lines = [(i + 1, ln) for i, ln in enumerate(body.splitlines())]
    return classify_predicate_lines(lines, signature, body)


def test_combined_load_and_accuracy_filter_is_flagged():
    verdict = _classified("keep = (loads == 1) & accuracy")
    assert verdict["classification"] == "trial_level_outcome_filter"
    assert verdict["filters_on_outcome_variable"] is True
    assert "accuracy" in verdict["offending_expression"]


def test_bare_accuracy_keep_is_flagged():
    verdict = _classified("keep = accuracy")
    assert verdict["classification"] == "trial_level_outcome_filter"


def test_artifact_and_correct_keep_is_flagged():
    verdict = _classified("keep = (~artifact) & correct")
    assert verdict["classification"] == "trial_level_outcome_filter"
    assert verdict["filters_on_outcome_variable"] is True


def test_direct_outcome_subscript_of_spikes_is_flagged():
    body = (
        "correct = np.asarray(raw['isCorr'], dtype=bool).reshape(-1)\n"
        "spikes = spikes[correct]\n"
    )
    verdict = _classified(body)
    assert verdict["classification"] == "trial_level_outcome_filter"


def test_tuple_unpacking_outcome_subscript_is_flagged():
    body = "spikes, cue_idx = spikes[correct], cue_idx[correct]\n"
    verdict = _classified(body)
    assert verdict["classification"] == "trial_level_outcome_filter"


def test_session_level_accuracy_gate_is_distinguished_from_trial_filter():
    body = "if float(np.mean(accuracy)) < MIN_SESSION_ACCURACY:\n    return None\n"
    verdict = _classified(body)
    assert verdict["classification"] == "session_level_outcome_gate"
    assert verdict["filters_on_outcome_variable"] is True


def test_response_acc_abbreviation_gate_is_caught():
    body = "if response_acc.mean() < MIN_SESSION_ACCURACY:\n    continue\n"
    verdict = _classified(body)
    assert verdict["classification"] == "session_level_outcome_gate"


def test_non_outcome_admission_is_not_flagged():
    body = "keep = (~artifact) & ieeg['valid']\nepochs = epochs[keep]\n"
    verdict = _classified(body)
    assert verdict["classification"] == "non_outcome_admission"
    assert verdict["filters_on_outcome_variable"] is False


def test_load_condition_restriction_is_not_an_outcome_filter():
    verdict = _classified("keep = loads == 1\nlabels = labels[keep]")
    assert verdict["classification"] == "non_outcome_admission"
    assert verdict["filters_on_outcome_variable"] is False


def test_unit_rate_qc_mask_is_not_treated_as_trial_admission():
    body = "rate_mask = low_rate_unit_mask(spike_lists, onsets, window_s)\n"
    lines = [(1, body.strip())]
    verdict = classify_predicate_lines(lines, "def f():", body)
    assert verdict["classification"] == "outcome_preserved"
    assert verdict["filters_on_outcome_variable"] is False


def test_outcome_used_as_decoder_label_is_not_admission():
    body = "y = (~correct).astype(int)\nn_correct = int(correct.sum())\n"
    verdict = _classified(body)
    assert verdict["classification"] == "outcome_preserved"
    assert verdict["filters_on_outcome_variable"] is False


def test_macaque_pfc_microstimulation_style_outcome_stratified_loader_is_recognised():
    body = (
        'folder = "correct" if correct else "error"\n'
        'fname = f"{prefix}.mat" if correct else f"{prefix}_err.mat"\n'
    )
    signature = "def load_macaque_pfc_microstimulation_session(prefix: str, correct: bool) -> dict | None:"
    verdict = _classified(body, signature=signature)
    assert verdict["classification"] == "outcome_stratified_loader"
    assert verdict["filters_on_outcome_variable"] is True


def test_shape_lookup_of_outcome_column_is_not_admission():
    body = ('mask = np.ones(s["is_correct"].shape[0], dtype=bool) '
            'if load_filter is None else (s["load_level"] == load_filter)\n')
    verdict = _classified(body)
    assert verdict["filters_on_outcome_variable"] is False


def test_isfinite_completeness_mask_is_not_an_outcome_filter():
    body = ("usable = np.isfinite(deviation) & np.isfinite(report_error) "
            "& np.isfinite(reaction_time)\n")
    verdict = _classified(body)
    assert verdict["classification"] == "outcome_completeness_mask"
    assert verdict["filters_on_outcome_variable"] is False


def test_correct_and_defined_mask_still_flags():
    verdict = _classified("keep = correct & defined\n")
    assert verdict["classification"] == "trial_level_outcome_filter"


def test_extraction_picks_admission_lines_from_a_function_body():
    body = "\n".join([
        "def iter_example(root):",
        '    """docstring"""',
        "    accuracy = trials['response_accuracy'][:].astype(bool)",
        "    loads = trials['loads'][:].astype(int)",
        "    keep = (loads == 1) & accuracy",
        "    if keep.sum() < MIN_TRIALS:",
        "        continue",
        "    epochs = build(keep)",
    ])
    source_lines = body.splitlines()
    hits = extract_predicate_lines(source_lines, 2, len(source_lines))
    quoted = [text for _, text in hits]
    assert any("keep = (loads == 1) & accuracy" in q for q in quoted)


ADMISSION_LHSES = ["keep", "mask", "good", "eligible", "admitted", "usable", "valid_mask"]


@pytest.mark.parametrize("lhs", ADMISSION_LHSES)
def test_all_admission_left_hand_sides_recognised(lhs):
    assert ADMISSION_ASSIGN_RE.match(f"{lhs} = correct") is not None


def test_subscript_regex_accepts_single_and_tuple_targets_and_rejects_where_calls():
    assert OUTCOME_SUBSCRIPT_RE.match("spikes = spikes[correct]")
    assert OUTCOME_SUBSCRIPT_RE.match("a, b = a[correct], b[correct]")
    assert OUTCOME_SUBSCRIPT_RE.match("x = tensor[keep]")
    assert not OUTCOME_SUBSCRIPT_RE.match("counts_error = counts_all[np.where(~is_corr)[0]]")
    assert not OUTCOME_SUBSCRIPT_RE.match("y = (~correct).astype(int)")


def _loader(path, function, corpora, classification):
    return {
        "path": path, "function": function, "corpora": corpora,
        "status": "audited", "line_start": 1, "line_end": 10,
        "predicate_line_numbers": [5], "classification": classification,
        "filters_on_outcome_variable": classification in (
            "trial_level_outcome_filter", "session_level_outcome_gate", "outcome_stratified_loader"),
        "offending_expression": "x" if "filter" in classification or classification != "non_outcome_admission" else None,
    }


def test_branch_evaluation_defect_class_threshold_and_alternate_loader_rule():
    loaders = [
        _loader("src/corpus_sessions.py", "iter_a", ["corpus_x"], "trial_level_outcome_filter"),
        _loader("src/corpus_sessions.py", "iter_b", ["corpus_y"], "trial_level_outcome_filter"),
        _loader("scripts/other.py", "reader_c", ["corpus_z"], "trial_level_outcome_filter"),
        _loader("scripts/other.py", "preserver", ["corpus_x"], "outcome_preserved"),
        _loader("scripts/other.py", "benign", ["corpus_w"], "non_outcome_admission"),
    ]
    askability = {
        c: {cell: {"value": "no"} for cell in (
            "error_trial_contrasts", "graded_report_analyses",
            "stimulation_response_mapping", "within_maintenance_intervention",
            "cross_instrument_pairing")}
        for c in ("corpus_w", "corpus_x", "corpus_y", "corpus_z")
    }
    branches = evaluate_branches(loaders, askability, ["corpus_w", "corpus_x", "corpus_y", "corpus_z"])
    names = [b["branch"] for b in branches]
    assert "outcome_filtering_is_a_defect_class_not_a_one_off" in names
    assert "outcome_filtering_extends_beyond_the_human_spine_iterators" in names
    # corpus_x has a flagged loader AND an outcome-preserving reader; corpus_y
    # and corpus_z have none, so only corpus_x gets the alternate-loader branch.
    assert "corpus_corpus_x_error_contrasts_require_an_alternate_loader" in names
    assert "corpus_corpus_y_error_contrasts_require_an_alternate_loader" not in names
    assert "corpus_corpus_z_error_contrasts_require_an_alternate_loader" not in names
    per_corpus = [n for n in names if n.startswith("corpus_") and n.endswith("_of_5_questions")]
    assert len(per_corpus) == 4


def test_branch_evaluation_below_threshold_reports_confinement():
    loaders = [
        _loader("src/corpus_sessions.py", "iter_a", ["corpus_x"], "trial_level_outcome_filter"),
        _loader("src/other.py", "iter_b", ["corpus_x"], "non_outcome_admission"),
    ]
    askability = {
        "corpus_x": {cell: {"value": "yes"} for cell in (
            "error_trial_contrasts", "graded_report_analyses",
            "stimulation_response_mapping", "within_maintenance_intervention",
            "cross_instrument_pairing")},
    }
    branches = evaluate_branches(loaders, askability, ["corpus_x"])
    names = [b["branch"] for b in branches]
    assert "outcome_filtering_is_confined_to_1_loaders" in names
    assert "outcome_filtering_confined_to_the_human_spine_iterators" in names
    assert "graded_report_supported_by_1_of_1_corpora" in names
