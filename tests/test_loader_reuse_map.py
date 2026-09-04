"""Tests for scripts/build_loader_reuse_map.py: every corpus entry must
declare a valid loader-status category, and the summary counts must equal
what a fresh count over the map itself produces (so the two cannot drift
apart the way a hand-typed summary elsewhere in this project already did
once)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_loader_reuse_map as mod  # noqa: E402

VALID_STATUSES = {"yes", "cache_only", "no", "not_applicable", "unknown_not_confirmed"}


def test_every_corpus_declares_a_valid_loader_status():
    for name, entry in mod.CORPUS_LOADER_MAP.items():
        assert entry["has_tensor_loader"] in VALID_STATUSES, name


def test_every_yes_entry_names_a_script_and_function():
    for name, entry in mod.CORPUS_LOADER_MAP.items():
        if entry["has_tensor_loader"] == "yes":
            assert entry.get("script"), name
            assert entry.get("function"), name


def test_summary_counts_match_a_fresh_count_over_the_map():
    payload = mod.build_loader_reuse_map()
    fresh_yes = sum(1 for v in mod.CORPUS_LOADER_MAP.values() if v["has_tensor_loader"] == "yes")
    assert payload["n_with_a_working_tensor_loader"] == fresh_yes
    assert payload["n_corpora"] == len(mod.CORPUS_LOADER_MAP)
