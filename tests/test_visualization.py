"""Tests for src/visualization.py."""

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import visualization as viz
from visualization import (
    nature_style,
    save_figure,
    PALETTE,
    fig1_dataset_and_load_effect,
    fig2_manifold_geometry,
)


class TestNatureStyle:
    def test_sets_font_size(self):
        nature_style()
        assert matplotlib.rcParams["font.size"] == 7

    def test_disables_top_right_spines(self):
        nature_style()
        assert matplotlib.rcParams["axes.spines.right"] is False
        assert matplotlib.rcParams["axes.spines.top"] is False


class TestPalette:
    def test_two_condition_keys_present(self):
        for key in ["zero_back", "one_back", "two_back", "correct", "error"]:
            assert key in PALETTE

    def test_values_are_hex_colors(self):
        for v in PALETTE.values():
            assert v.startswith("#")
            assert len(v) == 7


class TestSaveFigure:
    def test_writes_pdf_and_png(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            monkeypatch.setattr(viz, "FIGURES_DIR", Path(tmp))
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            save_figure(fig, "test_fig")
            assert (Path(tmp) / "test_fig.pdf").exists()
            assert (Path(tmp) / "test_fig.png").exists()
            plt.close(fig)


class TestFig1DatasetAndLoadEffect:
    def test_returns_figure_without_error(self, rng):
        N, T, C = 30, 50, 6
        epochs_z = rng.standard_normal((N, T, C)).astype(np.float32)
        times = np.linspace(-0.2, 1.5, T)
        task_id = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int8)
        fig = fig1_dataset_and_load_effect(epochs_z, times, task_id)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFig2ManifoldGeometry:
    def test_returns_figure_without_error(self, rng):
        pr_by_load = {0: rng.uniform(2, 6, 20), 1: rng.uniform(2, 6, 20), 2: rng.uniform(2, 6, 20)}
        theta = rng.uniform(0, np.pi / 2, (40, 4))
        times = np.linspace(-0.2, 1.5, 40)
        fig = fig2_manifold_geometry(pr_by_load, theta, times)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
