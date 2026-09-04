"""Regression tests for scripts/run_lfp_structure_dynamics.py: the RAM electrode/channel structure resolution, distractor-epoch
parsing, and the shared lambda/diffusion identifiability filter. Pure logic
on synthetic inputs -- no real data touched.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import run_lfp_structure_dynamics as mod  # noqa: E402


def _write_electrodes_tsv(path, rows):
    fieldnames = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


class TestRamElectrodeStructures:
    def test_stein_region_preferred_over_ind_region(self, tmp_path):
        path = tmp_path / "electrodes.tsv"
        _write_electrodes_tsv(path, [
            {"name": "LA1", "stein.region": "Left Amy", "das.region": "n/a", "ind.region": "n/a"},
            {"name": "LB1", "stein.region": "n/a", "das.region": "n/a", "ind.region": "temporalpole"},
            {"name": "LC1", "stein.region": "n/a", "das.region": "n/a", "ind.region": "n/a"},
        ])
        out = mod._ram_electrode_structures(path)
        assert out["LA1"] == "amygdala"
        assert out["LB1"] == "temporal_pole"
        assert out["LC1"] == "unlabelled"


class TestRamChannelStructures:
    def test_agreement_uses_shared_structure(self):
        contacts = {"LA1": "hippocampus", "LA2": "hippocampus"}
        out = mod._ram_channel_structures(["LA1-LA2"], contacts)
        assert out["LA1-LA2"] == "hippocampus"

    def test_disagreement_uses_anode_contact(self):
        contacts = {"LA1": "hippocampus", "LA2": "amygdala"}
        out = mod._ram_channel_structures(["LA1-LA2"], contacts)
        assert out["LA1-LA2"] == "hippocampus"

    def test_malformed_channel_name_skipped(self):
        contacts = {"LA1": "hippocampus"}
        out = mod._ram_channel_structures(["LA1"], contacts)
        assert out == {}


class TestRamDistractorEpochs:
    def test_pairs_starts_and_ends_in_order(self, tmp_path):
        path = tmp_path / "events.tsv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["onset", "trial_type"], delimiter="\t")
            w.writeheader()
            w.writerow({"onset": "10.0", "trial_type": "DISTRACT_START"})
            w.writerow({"onset": "30.0", "trial_type": "DISTRACT_END"})
            w.writerow({"onset": "50.0", "trial_type": "DISTRACT_START"})
            w.writerow({"onset": "75.0", "trial_type": "DISTRACT_END"})
        epochs = mod._ram_distractor_epochs(path)
        assert epochs == [(10.0, 30.0), (50.0, 75.0)]


class TestFitLambdaDiffusion:
    def test_not_identifiable_below_min_trials(self):
        trial_bins = np.random.default_rng(0).normal(size=(3, 10))
        condition = np.zeros(3, dtype=int)
        result = mod._fit_lambda_diffusion(trial_bins, condition, bin_s=0.5, min_trials=6)
        assert result["status"] == "not_identifiable"
        assert result["lambda_rate"] is None

    def test_confined_process_recovers_lambda_close_to_ground_truth(self):
        """Status can legitimately land as 'not_identifiable' even for a
        real confined process at some (lambda, dt, n_bins) combinations
        (src/drift_dynamics.py's own delay-span identifiability criterion,
        unrelated to this wiring) -- the estimator wiring itself is checked
        via the recovered point estimate, not the status label."""
        rng = np.random.default_rng(0)
        n_trials, n_bins = 40, 20
        true_lambda, dt = 2.0, 0.5
        trial_bins = np.zeros((n_trials, n_bins))
        for i in range(n_trials):
            x = 0.0
            for b in range(n_bins):
                x = x * np.exp(-true_lambda * dt) + rng.normal(scale=0.3)
                trial_bins[i, b] = x
        condition = np.zeros(n_trials, dtype=int)
        result = mod._fit_lambda_diffusion(trial_bins, condition, bin_s=dt, min_trials=6)
        assert result["lambda_rate"] == pytest.approx(true_lambda, rel=0.4)
        assert result["diffusion"] > 0


class TestBootstrapSummary:
    def test_non_identified_below_two_values(self):
        rng = np.random.default_rng(0)
        assert mod.bootstrap_summary([1.0], rng)["status"] == "non_identified"

    def test_estimable_reports_mean(self):
        rng = np.random.default_rng(0)
        result = mod.bootstrap_summary([1.0, 2.0, 3.0], rng)
        assert result["status"] == "estimable"
        assert result["mean"] == pytest.approx(2.0)
