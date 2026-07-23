"""
visualization.py — Publication-grade figure utilities.

All figures follow Nature Neuroscience formatting conventions:
  - 89 mm wide (single column) or 183 mm (double column)
  - Helvetica/Arial typeface, 7 pt minimum font size
  - 300 dpi minimum for raster exports

References
----------
Nature Neuroscience Guide to Authors (2024): figure specifications.
Rougier NP et al. (2014) Ten Simple Rules for Better Figures. PLoS Comput Biol.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from pathlib import Path
from numpy.typing import NDArray

FIGURES_DIR = Path(__file__).parent.parent / "figures"
_JITTER_RNG = np.random.default_rng(0)  # deterministic scatter jitter across figure regens

# ── Nature style ────────────────────────────────────────────────────────────────

def nature_style() -> None:
    """Apply Nature Neuroscience–compatible matplotlib rcParams."""
    rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2,
        "ytick.major.size": 2,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.right": False,
        "axes.spines.top": False,
    })


def save_figure(fig: plt.Figure, name: str, formats: list[str] | None = None) -> None:
    """Save figure to figures/ in both PDF and PNG formats.

    Parameters
    ----------
    name    : filename without extension
    formats : list of extensions; defaults to ['pdf', 'png']
    """
    FIGURES_DIR.mkdir(exist_ok=True)
    if formats is None:
        formats = ["pdf", "png"]
    for fmt in formats:
        fig.savefig(FIGURES_DIR / f"{name}.{fmt}", bbox_inches="tight")


# ── Colour palette ──────────────────────────────────────────────────────────────

PALETTE = {
    "zero_back":       "#4E79A7",
    "one_back":        "#59A14F",
    "two_back":        "#E15759",
    "target":          "#F28E2B",
    "non_target":      "#76B7B2",
    "correct":         "#59A14F",
    "error":           "#E15759",
    "neutral":         "#BAB0AC",
}


# ── Figure 1: Dataset & load effect ────────────────────────────────────────────

def fig1_dataset_and_load_effect(
    epochs_z: NDArray,
    times: NDArray,
    task_id: NDArray,
    ch_idx: int = 5,
    example_chs: list[int] | None = None,
) -> plt.Figure:
    """Figure 1 — Data overview and WM load replication.

    Panels:
      A: Mean high-gamma timecourse per condition (3 example channels)
      B: WM load effect: mean maintenance HGP vs N-back level (bar + scatter)
      C: 3D PCA trajectory overview (3 conditions, 20 trials each)
    """
    nature_style()
    if example_chs is None:
        example_chs = list(range(min(3, epochs_z.shape[2])))

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.45)

    # ── A: ERP by condition ──
    ax_a = fig.add_subplot(gs[0])
    maint_shade = (0.3, 1.4)
    for cond, key in zip([0, 1, 2], ["zero_back", "one_back", "two_back"]):
        mask = task_id == cond
        if mask.sum() == 0:
            continue
        m = epochs_z[mask][:, :, example_chs[0]].mean(axis=0)
        s = epochs_z[mask][:, :, example_chs[0]].std(axis=0) / np.sqrt(mask.sum())
        ax_a.plot(times, m, color=PALETTE[key], label=f"{cond}-back")
        ax_a.fill_between(times, m - s, m + s, color=PALETTE[key], alpha=0.2)
    ax_a.axvspan(*maint_shade, color="gold", alpha=0.12, zorder=0)
    ax_a.axvline(0, color="k", lw=0.5, ls="--")
    ax_a.axhline(0, color="k", lw=0.3)
    ax_a.set_xlabel("Time from stimulus onset (s)")
    ax_a.set_ylabel("HGP (z-score)")
    ax_a.set_title("A  High-gamma by load", loc="left", fontweight="bold")
    ax_a.legend(frameon=False)

    # ── B: Maintenance HGP by load ──
    ax_b = fig.add_subplot(gs[1])
    maint_mask = (times >= maint_shade[0]) & (times <= maint_shade[1])
    load_means, load_sems = [], []
    for cond in [0, 1, 2]:
        mask = task_id == cond
        if mask.sum() == 0:
            load_means.append(0); load_sems.append(0)
            continue
        vals = epochs_z[mask][:, maint_mask, :].mean(axis=(1, 2))
        load_means.append(vals.mean())
        load_sems.append(vals.std() / np.sqrt(len(vals)))
        jitter = 0.1 * _JITTER_RNG.standard_normal(len(vals))
        ax_b.scatter(
            np.full(len(vals), cond) + jitter, vals,
            color=PALETTE[["zero_back", "one_back", "two_back"][cond]],
            alpha=0.4, s=4, zorder=3
        )
    ax_b.bar(
        [0, 1, 2], load_means, yerr=load_sems,
        color=[PALETTE["zero_back"], PALETTE["one_back"], PALETTE["two_back"]],
        width=0.5, alpha=0.7, capsize=3, error_kw={"lw": 1}
    )
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["0-back", "1-back", "2-back"])
    ax_b.set_ylabel("Mean HGP (z-score, maintenance)")
    ax_b.set_title("B  Load effect", loc="left", fontweight="bold")

    # ── C: 3D PCA trajectories ──
    ax_c = fig.add_subplot(gs[2], projection="3d")
    N, T, C = epochs_z.shape
    X_flat = epochs_z.reshape(-1, C)
    Xc = X_flat - X_flat.mean(0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = (Xc @ Vt[:3].T).reshape(N, T, 3)

    for cond, key in zip([0, 1, 2], ["zero_back", "one_back", "two_back"]):
        mask = np.where(task_id == cond)[0][:15]
        for ti in mask:
            tr = scores[ti]
            ax_c.plot(tr[:, 0], tr[:, 1], tr[:, 2],
                      color=PALETTE[key], lw=0.5, alpha=0.4)

    ax_c.set_xlabel("PC1", fontsize=5)
    ax_c.set_ylabel("PC2", fontsize=5)
    ax_c.set_zlabel("PC3", fontsize=5)
    ax_c.set_title("C  Neural trajectories", loc="left", fontweight="bold")

    return fig


# ── Figure 2: Participation ratio & subspace geometry ──────────────────────────

def fig2_manifold_geometry(
    pr_by_load: dict,
    theta_tgt_vs_ntgt: NDArray,
    times: NDArray,
    sig_clusters: list[dict] | None = None,
) -> plt.Figure:
    """Figure 2 — Intrinsic dimensionality and principal angles.

    Panels:
      A: PR distribution per N-back load (violin + mean ± CI)
      B: Time-resolved θ_min: target vs non-target in 2-back
    """
    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # ── A: PR by load ──
    ax = axes[0]
    positions = [0, 1, 2]
    for i, (cond, key) in enumerate(zip([0, 1, 2], ["zero_back", "one_back", "two_back"])):
        pr = pr_by_load.get(cond, np.array([]))
        if len(pr) == 0:
            continue
        parts = ax.violinplot([pr], positions=[i], widths=0.6, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(PALETTE[key])
            pc.set_alpha(0.6)
        ax.scatter(
            np.full(len(pr), i) + 0.1 * _JITTER_RNG.standard_normal(len(pr)),
            pr, s=5, color=PALETTE[key], alpha=0.5, zorder=3
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(["0-back", "1-back", "2-back"])
    ax.set_ylabel("Participation ratio (a.u.)")
    ax.set_title("A  Intrinsic dimensionality", loc="left", fontweight="bold")

    # ── B: Principal angles timecourse ──
    ax = axes[1]
    if theta_tgt_vs_ntgt is not None and len(theta_tgt_vs_ntgt) > 0:
        theta_min = theta_tgt_vs_ntgt[:, 0] if theta_tgt_vs_ntgt.ndim == 2 else theta_tgt_vs_ntgt
        ax.plot(times[:len(theta_min)], np.degrees(theta_min), color=PALETTE["two_back"], lw=1)
        ax.axhline(90, color="k", lw=0.5, ls="--", alpha=0.5, label="Orthogonal (90°)")
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.5, label="Parallel (0°)")
        ax.axvspan(0.3, 1.4, color="gold", alpha=0.12, zorder=0, label="Maintenance")

        if sig_clusters:
            for c in sig_clusters:
                ax.axvspan(
                    c["start_s"], c["end_s"],
                    color="purple", alpha=0.15, zorder=1
                )
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Minimum principal angle (°)")
    ax.set_title("B  Subspace angle: target vs non-target", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig


# ── Figure 3: Trajectory tangling ──────────────────────────────────────────────

def fig3_tangling(
    Q_tgt: NDArray,
    Q_ntgt: NDArray,
    times: NDArray,
    ci_tgt: tuple | None = None,
    ci_ntgt: tuple | None = None,
) -> plt.Figure:
    """Figure 3 — Trajectory tangling Q(t) by trial condition.

    Panel A: Mean Q(t) timecourse, target vs. non-target (2-back)
    Panel B: Peak Q in maintenance window, violin plot
    """
    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # ── A: Timecourse ──
    ax = axes[0]
    T_len = min(len(Q_tgt), len(times))
    ax.plot(times[:T_len], Q_tgt[:T_len], color=PALETTE["target"], lw=1, label="Target")
    ax.plot(times[:T_len], Q_ntgt[:T_len], color=PALETTE["non_target"], lw=1, label="Non-target")

    if ci_tgt is not None:
        lo, hi = ci_tgt
        ax.fill_between(times[:T_len], lo[:T_len], hi[:T_len],
                        color=PALETTE["target"], alpha=0.2)
    if ci_ntgt is not None:
        lo, hi = ci_ntgt
        ax.fill_between(times[:T_len], lo[:T_len], hi[:T_len],
                        color=PALETTE["non_target"], alpha=0.2)

    ax.axvspan(0.3, 1.4, color="gold", alpha=0.12, zorder=0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tangling Q(t)")
    ax.set_title("A  Trajectory tangling timecourse", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    # ── B: Peak Q violin ──
    ax = axes[1]
    groups = {"Target": Q_tgt, "Non-target": Q_ntgt}
    colors = [PALETTE["target"], PALETTE["non_target"]]
    parts = ax.violinplot(
        [g for g in groups.values()],
        positions=[0, 1], widths=0.6, showmedians=True
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("Peak Q (maintenance window)")
    ax.set_title("B  Peak tangling by condition", loc="left", fontweight="bold")

    plt.tight_layout()
    return fig


# ── Figure 4: DMD eigenspectrum ────────────────────────────────────────────────

def fig4_eigenspectrum(
    evals_correct: NDArray,
    evals_error: NDArray,
) -> plt.Figure:
    """Figure 4 — DMD eigenspectrum: stability comparison.

    Panel A: Unit circle plot, eigenvalue scatter per condition
    Panel B: |λ| distribution (violin) per condition
    """
    nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    theta = np.linspace(0, 2 * np.pi, 200)

    # ── A: Unit circle ──
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.7, alpha=0.5, label="Unit circle")
    for evals, key, label in [
        (evals_correct, "non_target", "2-back non-target"),
        (evals_error, "target", "2-back target"),
    ]:
        evals_flat = evals.ravel() if evals.ndim > 1 else evals
        ax.scatter(
            evals_flat.real, evals_flat.imag,
            color=PALETTE[key], alpha=0.3, s=4, label=label
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title("A  DMD eigenspectrum", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    # ── B: |λ| distribution ──
    ax = axes[1]
    groups = {
        "Non-target": np.abs(evals_correct.ravel()),
        "Target": np.abs(evals_error.ravel()),
    }
    colors = [PALETTE["non_target"], PALETTE["target"]]
    parts = ax.violinplot(list(groups.values()), positions=[0, 1],
                          widths=0.6, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    ax.axhline(1.0, color="k", lw=0.7, ls="--", label="|λ|=1 (neutral)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel("|λ| (eigenvalue modulus)")
    ax.set_title("B  Stability by condition", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig


# ── Figure 5: LQR rescue ───────────────────────────────────────────────────────

def fig5_lqr_rescue(
    x_fail: NDArray,
    x_rescue: NDArray,
    u_traj: NDArray,
    pareto: dict | None = None,
    capacity: dict | None = None,
) -> plt.Figure:
    """Figure 5 — Control-theoretic rescue and BCI feasibility.

    Panel A: Failing vs. rescued trajectory (2D projection)
    Panel B: Control signal u(t)
    Panel C: Energy–accuracy Pareto curve
    Panel D: Electrode capacity curve
    """
    nature_style()
    n_panels = 2 + (pareto is not None) + (capacity is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(7.2, 2.8))
    axes = np.array(axes).ravel()

    # ── A: Trajectories ──
    ax = axes[0]
    ax.plot(x_fail[:, 0], x_fail[:, 1], color=PALETTE["error"], lw=1.2,
            label="Failing trajectory")
    ax.plot(x_rescue[:, 0], x_rescue[:, 1], color=PALETTE["correct"], lw=1.2,
            ls="--", label="Rescued (LQR)")
    ax.scatter(*x_fail[0, :2], color=PALETTE["error"], s=20, zorder=5)
    ax.scatter(*x_rescue[-1, :2], color=PALETTE["correct"], s=20,
               marker="*", zorder=5)
    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title("A  LQR rescue", loc="left", fontweight="bold")
    ax.legend(frameon=False)

    # ── B: Control signal ──
    ax = axes[1]
    t_ctrl = np.arange(len(u_traj))
    for ch in range(min(u_traj.shape[1], 4)):
        ax.plot(t_ctrl, u_traj[:, ch], lw=0.8, alpha=0.8, label=f"u{ch+1}")
    ax.axhline(0, color="k", lw=0.3)
    ax.set_xlabel("Step")
    ax.set_ylabel("Control u(t)")
    ax.set_title("B  Optimal control signal", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2)

    # ── C: Pareto ──
    if pareto is not None:
        ax = axes[2]
        ax.scatter(pareto["energies"], pareto["errors"],
                   c=pareto["q_values"], cmap="viridis", s=30, zorder=3)
        ax.set_xlabel("Control energy (Σu²)")
        ax.set_ylabel("Final state error ‖x(T)-x_f‖")
        ax.set_title("C  Energy–accuracy Pareto", loc="left", fontweight="bold")

    # ── D: Electrode capacity ──
    if capacity is not None:
        ax = axes[-(1 if pareto is None else 0) - 1 + n_panels]
        n_ch = capacity["n_channels"]
        m = capacity["pr_contrast_mean"]
        s = capacity["pr_contrast_std"]
        ax.fill_between(n_ch, m - s, m + s, alpha=0.3, color=PALETTE["two_back"])
        ax.plot(n_ch, m, "o-", color=PALETTE["two_back"], lw=1)
        ax.set_xlabel("Number of electrodes")
        ax.set_ylabel("PR contrast (2-back − 0-back)")
        ax.set_title("D  Electrode capacity", loc="left", fontweight="bold")

    plt.tight_layout()
    return fig
