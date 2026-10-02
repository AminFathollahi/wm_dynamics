import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import build_analysis_freeze as freeze  # noqa: E402
from provenance import sha256_file  # noqa: E402

LOCK_CORPORA = sorted(json.loads((ROOT / "provenance" / "data_lock.json").read_text())["corpora"])


@pytest.fixture(scope="module")
def cells():
    return freeze.build_cells()


def synthetic_co_primary(extra_candidates=None):
    rows = []
    for key, (corpus, region_cell) in freeze.BENCHMARK_CELLS.items():
        for candidate in freeze.DESIGN_SPACES[key]:
            if candidate != freeze.DPCA:
                rows.append({"corpus": corpus, "region_cell": region_cell, "candidate": candidate, "co_primary": True})
        rows.append({"corpus": corpus, "region_cell": region_cell, "candidate": "factor_analysis_other", "co_primary": False})
        rows.append({"corpus": corpus, "region_cell": region_cell, "candidate": "unscored", "co_primary": None})
    rows.extend(extra_candidates or [])
    return {"co_primary": rows}


def test_every_data_lock_corpus_and_every_link_is_present(cells):
    for corpus in LOCK_CORPORA:
        assert {c["link"] for c in cells if c["corpus"] == corpus} == set(range(1, 13)), corpus
    wolff = [c for c in cells if c["corpus"] == "wolff_eeg_impulse" and c["link"] == 6]
    assert {c["experiment"] for c in wolff} == {"experiment_1", "experiment_2"}
    assert len(cells) == len(freeze.HYPOTHESIS_LINKS) * (len(LOCK_CORPORA) + 1)


def test_every_cell_is_frozen_or_refused_with_a_reason(cells):
    assert {c["status"] for c in cells} == {"frozen", "not_askable"}
    assert all(isinstance(c["reason"], str) and c["reason"].strip() for c in cells if c["status"] == "not_askable")


def test_every_frozen_cell_is_complete_and_its_detectable_correlation_follows_the_cluster_count(cells):
    frozen = [c for c in cells if c["status"] == "frozen"]
    assert frozen
    for c in frozen:
        assert c["test"] and c["outcome"] is not None and c["unit_set"]["primary"], c["id"]
        assert c["cluster_n"] is not None and c["cluster_unit"], c["id"]
        mdc = c["minimum_detectable_correlation"]
        assert mdc["status"] == ("estimable" if c["cluster_n"] >= 4 else "not_estimable"), c["id"]
        assert (mdc["trial_bound"] is None) == (mdc["n_trials"] is None) and (mdc["n_trials"] is not None or mdc["trial_bound_reason"]), c["id"]
        if c["link"] in freeze.LINKS_USING_POPULATION_STATE:
            assert c["state_spaces"] and all(c["state_spaces"].values()), c["id"]
        if (c["hypothesis"], c["link"]) in freeze.OUTCOME_CELLS:
            assert c["outcome"]["primary"], c["id"]


def test_stimulation_cells_are_never_pooled_and_no_pooling_group_mixes_species_or_recording_types(cells):
    by_group = {}
    for c in cells:
        if c["link"] in freeze.STIMULATION_LINKS or c["link"] == 12 or c["hypothesis"] == "H7":
            assert c["pooling_group"] == "never_pooled", c["id"]
        elif c["pooling_group"] != "never_pooled":
            by_group.setdefault(c["pooling_group"], []).append(c)
    assert by_group
    for name, members in by_group.items():
        for key in ("species", "recording_type", "task_class", "link"):
            assert len({m[key] for m in members}) == 1, (name, key)
        assert members[0]["link"] in freeze.POOLABLE_LINKS


def test_state_space_lists_equal_the_co_primary_artifact_and_a_new_candidate_fails_loudly():
    artifact = synthetic_co_primary()
    derived = freeze.benchmark_state_spaces(artifact)
    for key, spaces in derived.items():
        assert spaces == sorted(freeze.DESIGN_SPACES[key])
    freeze.check_state_spaces_against_benchmark(artifact)
    artifact["co_primary"].append({"corpus": "dandi_000574", "region_cell": "pooled",
                                   "candidate": "temporal_diffusion_embedding", "co_primary": True})
    with pytest.raises(SystemExit):
        freeze.check_state_spaces_against_benchmark(artifact)


def test_delivered_artifact_agrees_with_the_design_and_the_written_freeze_hash_is_recorded():
    freeze.check_state_spaces_against_benchmark(json.loads(freeze.CO_PRIMARY_PATH.read_text()))
    if freeze.FREEZE_PATH.exists():
        assert sha256_file(freeze.FREEZE_PATH) in freeze.README_PATH.read_text()
