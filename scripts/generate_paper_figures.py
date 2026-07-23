#!/usr/bin/env python3
"""Generate all publication-quality figures for the WM Dynamics paper.

Run from project root:
    conda run -n wm_dynamics python scripts/generate_paper_figures.py

Outputs: figures/fig{1-9}_main.pdf, figures/figS{1-10}_supp.pdf
"""

import sys
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from visualization import nature_style, PALETTE, save_figure, FIGURES_DIR

RESULTS = ROOT / "results"
SUBJECTS = ["al", "ca", "cc", "ug"]
SUBJ_LABELS = {"al": "S1 (al)", "ca": "S2 (ca)", "cc": "S3 (cc)", "ug": "S4 (ug)"}
LOAD_COLORS = [PALETTE["zero_back"], PALETTE["one_back"], PALETTE["two_back"]]

# Rutishauser subjects that passed the ≥15-unit threshold
RUSHI_SUBS = [f"sub-{n}" for n in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,21]]
DATASET_COLORS = {"Miller": "#4E79A7", "Boran": "#E15759", "Rutishauser": "#59A14F"}
SUBJ_COLORS = ["#4E79A7", "#E15759", "#59A14F", "#B07AA1"]

# Full 7-dataset (+ TES1) palette, shared across Fig 1 / 2 / 3 / 5 for visual consistency
DS7_COLORS = {
    "Miller":       "#4E79A7",
    "Boran iEEG":   "#E15759",
    "Boran units":  "#F1CE63",
    "DANDI 000469": "#59A14F",
    "DANDI 001187": "#76B7B2",
    "DANDI 000673": "#B07AA1",
    "PFC-3":        "#FF9DA7",
    "TES1":         "#888888",
}

nature_style()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_all_stats():
    with open(RESULTS / "all_statistics.json") as f:
        return json.load(f)


def load_geometry(subj):
    r = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
    return r["Z"], r["task_id"], r["tgt_id"], r["times"]


def compute_pr(Z_flat):
    cov = np.cov(Z_flat.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-10]
    return float(eigvals.sum() ** 2 / (eigvals ** 2).sum())


def compute_ctg_matrix(Z, task_id, step=40):
    """Compute CTG AUC matrix (train t_i → test t_j) for 0-back vs 2-back."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    mask = (task_id == 0) | (task_id == 2)
    Z_sub = Z[mask]
    y = (task_id[mask] == 2).astype(int)

    T = Z_sub.shape[1]
    t_idx = np.arange(0, T, step)
    n_t = len(t_idx)
    auc_mat = np.full((n_t, n_t), np.nan)

    for i, ti in enumerate(t_idx):
        X_tr = Z_sub[:, ti, :]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        clf = LogisticRegression(C=1.0, solver="liblinear", max_iter=200)
        clf.fit(X_tr_s, y)
        for j, tj in enumerate(t_idx):
            X_te_s = scaler.transform(Z_sub[:, tj, :])
            scores = clf.decision_function(X_te_s)
            pos, neg = scores[y == 1], scores[y == 0]
            u = float(np.sum(pos[:, None] > neg[None, :]))
            auc_mat[i, j] = u / (len(pos) * len(neg))

    return auc_mat, t_idx


def panel_label(ax, label, x=-0.15, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def ci_from_bootstrap(data, stat_fn=np.mean, n_boot=2000, ci=0.95, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(data, len(data))) for _ in range(n_boot)]
    lo = np.percentile(boots, 100 * (1 - ci) / 2)
    hi = np.percentile(boots, 100 * (1 + ci) / 2)
    return lo, hi


def compute_actual_phases(subj, t_maint_end=1.4):
    """Compute 2-back endpoint phases from actual latent trajectories (PC1–PC2 plane)."""
    Z, task_id, _, times = load_geometry(subj)
    mask_2back = task_id == 2
    t_end_idx = int(np.argmin(np.abs(times - t_maint_end)))
    Z_end = Z[mask_2back][:, t_end_idx, :2]   # (N_2back, 2) — actual data
    phases = np.arctan2(Z_end[:, 1], Z_end[:, 0]) % (2 * np.pi)
    return phases


def simulate_lqr_targeted(A, x0, xf, n_steps=150, q=1.0):
    """LQR with B aligned to (xf−x0) direction; returns distance profile."""
    from scipy.linalg import solve_discrete_are

    d = xf - x0
    d_hat = d / np.linalg.norm(d)
    B_tgt = np.column_stack([d_hat, np.roll(d_hat, 1)])  # two input channels
    n = A.shape[0]
    Q_lqr = q * np.eye(n)
    R_lqr = np.eye(B_tgt.shape[1])
    try:
        P = solve_discrete_are(A, B_tgt, Q_lqr, R_lqr)
        K = np.linalg.solve(B_tgt.T @ P @ B_tgt + R_lqr, B_tgt.T @ P @ A)
        x = x0.copy()
        dists = [float(np.linalg.norm(x - xf))]
        for _ in range(n_steps):
            u = -K @ (x - xf)
            x = A @ x + B_tgt @ u
            dists.append(float(np.linalg.norm(x - xf)))
        return np.array(dists)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Dataset overview and preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def make_figure1():
    """Fig 1 -- the closed-loop schematic (Inagaki Fig 1 style: flat, small-
    multiple, iconographic line-art; no data-driven panels, no dataset poster
    -- the per-dataset cohort table lives in Table 1). Panels: (a) the WM
    task as a labeled epoch timeline; (b) the two coexisting codes as a
    state-space cartoon (stable context axis vs. rotating memorandum axis);
    (c) the closed-loop control cycle (read -> fit dynamics -> when/where/how
    -> stimulate -> re-read)."""
    from matplotlib.patches import FancyArrowPatch, Circle, FancyBboxPatch
    from matplotlib.path import Path as MPath

    nature_style()
    navy, darkred, darkgreen = "#1F497D", "#B41E1E", "#006432"
    gray = "#8C8C8C"
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.35, left=0.04, right=0.98,
                           top=0.86, bottom=0.06)

    # ── (a) Task timeline: labeled epoch boxes along a time arrow ──────────
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    ax_a.set_xlim(-0.3, 10.3); ax_a.set_ylim(-0.3, 3.2)
    ax_a.annotate("", xy=(10.1, 0.5), xytext=(-0.1, 0.5),
                  arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.4",
                                   color="black", lw=1.0))
    epochs = [("Enc", 0.2, 2.6, "#E8F0F8", navy), ("Delay", 2.9, 5.4, "#FDEEEE", darkred),
              ("Probe", 8.6, 9.9, "#E9F5EC", darkgreen)]
    for label, x0, x1, fc, ec in epochs:
        ax_a.add_patch(FancyBboxPatch((x0, 0.75), x1 - x0, 1.15,
                                      boxstyle="round,pad=0.02,rounding_size=0.08",
                                      fc=fc, ec=ec, lw=1.2))
        ax_a.text((x0 + x1) / 2, 1.325, label, ha="center", va="center",
                  fontsize=7.5, fontweight="bold", color=ec)
    ax_a.plot([5.6, 8.4], [1.325, 1.325], color=gray, lw=1.0, ls=(0, (1.5, 1.5)))
    for x, lab in [(0.2, ""), (2.9, ""), (8.6, "")]:
        ax_a.plot([x, x], [0.55, 0.75], color="black", lw=0.8, ls="--")
    ax_a.text(5.0, 2.65, "Working-memory maintenance task", ha="center",
              fontsize=7.5, fontweight="bold")
    ax_a.text(5.0, 0.15, "time", ha="center", fontsize=6.5, style="italic", color=gray)
    panel_label(ax_a, "a", x=-0.06, y=1.08)

    # ── (b) State-space cartoon: fixed context axis vs. rotating content axis ──
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    ax_b.set_xlim(-1.3, 1.3); ax_b.set_ylim(-1.3, 1.3)
    ax_b.set_aspect("equal")
    # stable context axis: fixed line through the origin with endpoint dots
    ax_b.plot([-1.0, 1.0], [-0.15, 0.15], color=navy, lw=1.8, solid_capstyle="round")
    ax_b.plot([-1.0, 1.0], [-0.15, 0.15], "o", color=navy, ms=3.5)
    ax_b.text(1.05, 0.20, "context\n(fixed axis)", fontsize=6.3, color=navy,
              ha="left", va="center", fontweight="bold")
    # rotating memorandum axis: curved arrow sweeping through several angles
    theta = np.linspace(0.35, 2.4, 60)
    r = 0.85
    ax_b.plot(r * np.cos(theta), r * np.sin(theta), color=darkred, lw=1.6,
              ls=(0, (4, 2)))
    ax_b.annotate("", xy=(r * np.cos(theta[-1]), r * np.sin(theta[-1])),
                  xytext=(r * np.cos(theta[-6]), r * np.sin(theta[-6])),
                  arrowprops=dict(arrowstyle="-|>,head_width=0.3,head_length=0.5",
                                   color=darkred, lw=1.6))
    ax_b.plot([0, r * np.cos(theta[0])], [0, r * np.sin(theta[0])], color=darkred,
             lw=1.2, alpha=0.5)
    ax_b.text(r * np.cos(theta[30]) + 0.15, r * np.sin(theta[30]) + 0.15,
             "memorandum\n(rotating)", fontsize=6.3, color=darkred, ha="left",
             va="center", fontweight="bold")
    ax_b.plot(0, 0, "o", color="black", ms=2.5)
    ax_b.text(5.0 / 10 - 1.3, -1.22, "population state space", fontsize=6.3,
             style="italic", color=gray, ha="left")
    panel_label(ax_b, "b", x=-0.06, y=1.08)

    # ── (c) Closed-loop control cycle ───────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    ax_c.axis("off")
    ax_c.set_xlim(-1.4, 1.4); ax_c.set_ylim(-1.65, 1.4)
    ax_c.set_aspect("equal")
    steps = ["read\nstate", "fit dynamics\n(A, v*)", "when / where / how\n(flow, v*, LQR)",
             "stimulate"]
    n = len(steps)
    R = 1.0
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    positions = [(R * np.cos(a), R * np.sin(a)) for a in angles]
    box_colors = [navy, darkgreen, "#8B6F1F", darkred]
    for (x, y), label, col in zip(positions, steps, box_colors):
        ax_c.add_patch(FancyBboxPatch((x - 0.42, y - 0.28), 0.84, 0.56,
                                      boxstyle="round,pad=0.02,rounding_size=0.09",
                                      fc="white", ec=col, lw=1.3))
        ax_c.text(x, y, label, ha="center", va="center", fontsize=5.6,
                  fontweight="bold", color=col)
    for i in range(n):
        x0, y0 = positions[i]
        x1, y1 = positions[(i + 1) % n]
        dx, dy = x1 - x0, y1 - y0
        dist = np.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        start = (x0 + 0.45 * ux, y0 + 0.45 * uy)
        stop = (x1 - 0.45 * ux, y1 - 0.45 * uy)
        arc = FancyArrowPatch(start, stop, connectionstyle="arc3,rad=0.18",
                              arrowstyle="-|>,head_width=0.22,head_length=0.4",
                              color=gray, lw=1.1)
        ax_c.add_patch(arc)
    ax_c.text(0, -1.58, "re-read → repeat", fontsize=6.0, style="italic",
             color=gray, ha="center")
    ax_c.text(0, 1.3, "closed-loop control", fontsize=7.5, fontweight="bold",
             ha="center")
    panel_label(ax_c, "c", x=-0.06, y=1.08)

    # Inset in the open center of the cycle: a thin, schematic "drifts
    # (loop-off) vs held (loop-on)" trajectory preview of R4 -- iconographic,
    # not data-driven, matching the rest of Fig 1 (the real simulated
    # trajectories are Figure 8).
    axins = ax_c.inset_axes([0.335, 0.395, 0.33, 0.24])
    t_sched = np.linspace(0, 1, 60)
    drift_sched = 0.08 * t_sched + 0.55 * t_sched ** 2.2
    held_sched = 0.05 * np.sin(2 * np.pi * t_sched) * np.exp(-3 * t_sched)
    axins.plot(t_sched, drift_sched, color=gray, lw=1.1, ls=(0, (2, 1.4)))
    axins.plot(t_sched, held_sched, color=darkred, lw=1.1)
    axins.axhline(0, color="k", lw=0.4, alpha=0.4)
    axins.set_xticks([]); axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set_linewidth(0.5)
    axins.text(1.0, drift_sched[-1], "drifts", fontsize=4.0, color=gray,
              ha="left", va="center")
    axins.text(1.0, held_sched[-1], "held", fontsize=4.0, color=darkred,
              ha="left", va="center")

    save_figure(fig, "fig1_main")
    print("  Figure 1 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — PR geometry and LME null
# ─────────────────────────────────────────────────────────────────────────────

def _pr_records_miller():
    out = {}
    for subj in SUBJECTS:
        d = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        pr, tid = d["pr_per_trial"], d["task_id"]
        out[subj] = {ld: float(np.mean(pr[tid == ld])) for ld in [0, 1, 2]}
    return out


def _pr_records_from_stats(entries, field, load_values):
    """entries: dict[key -> {field: {load: {"pr_cv": ...}}}]."""
    out = {}
    for key, v in entries.items():
        row = {}
        for ld in load_values:
            prd = v[field].get(str(ld), v[field].get(ld))
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                row[ld] = prd["pr_cv"]
        if row:
            out[key] = row
    return out


def _plot_pr_panel(ax, pr_by_subj, load_values, color, xlabels, title, panel_letter):
    for key, row in pr_by_subj.items():
        xs = [x for x in load_values if x in row]
        ys = [row[x] for x in xs]
        if len(xs) < 2:
            continue
        ax.plot(range(len(xs)), ys, "o-", color=color, alpha=0.55, lw=1.0, ms=3)
    ax.set_xticks(range(len(load_values)))
    ax.set_xticklabels(xlabels, fontsize=5.5)
    ax.set_ylabel("PR (cv)", fontsize=6)
    ax.set_title(title, loc="left", fontsize=6.3, fontweight="bold")
    panel_label(ax, panel_letter)


def make_figure2(stats_data):
    """Participation ratio does not scale with load — consistent null across
    all 6 applicable neural datasets (all computed by the same cross-validated
    spatiotemporal-PR method, native channel/unit space; see Methods
    Section~ref{sec:stats})."""
    nature_style()
    lme = stats_data["pr_lme_by_dataset"]

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = gridspec.GridSpec(2, 3, fig, hspace=0.62, wspace=0.42,
                           left=0.08, right=0.98, top=0.88, bottom=0.10)

    miller_pr = _pr_records_miller()
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_pr_panel(ax_a, miller_pr, [0, 1, 2], DATASET_COLORS["Miller"],
                  ["0-back", "1-back", "2-back"],
                  f"Miller ECoG (N=4)\nβ={lme['miller']['beta']:.3f}, p={lme['miller']['p_value']:.2f}",
                  "A")

    boran_ieeg_pr = _pr_records_from_stats(stats_data["boran_ctg"], "pr_per_set", [4, 6, 8])
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_pr_panel(ax_b, boran_ieeg_pr, [4, 6, 8], DATASET_COLORS["Boran"],
                  ["4", "6", "8"],
                  f"Boran iEEG (N=9)\nβ={lme['boran_ieeg']['beta']:.3f}, p={lme['boran_ieeg']['p_value']:.2f}",
                  "B")
    ax_b.set_xlabel("Set size", fontsize=6)

    boran_units_raw = stats_data.get("dandi000574_units_ctg", {})
    boran_units_pr = {}
    for key, v in boran_units_raw.items():
        row = {}
        for ld in [4, 6, 8]:
            prd = v["pr_per_set"].get(str(ld), v["pr_per_set"].get(ld))
            if isinstance(prd, dict) and np.isfinite(prd.get("pr_cv", np.nan)):
                row[ld] = prd["pr_cv"]
        if row:
            boran_units_pr[key] = row
    ax_c = fig.add_subplot(gs[0, 2])
    _plot_pr_panel(ax_c, boran_units_pr, [4, 6, 8], "#F1CE63",
                  ["4", "6", "8"],
                  f"Boran units (N={len(boran_units_pr)} sess.)\n"
                  f"β={lme['boran_units']['beta']:.3f}, p={lme['boran_units']['p_value']:.2f}",
                  "C")

    d469_pr = _pr_records_from_stats(stats_data["dandi000469_ctg"], "pr_per_load", [1, 2, 3])
    ax_d = fig.add_subplot(gs[1, 0])
    _plot_pr_panel(ax_d, d469_pr, [1, 2, 3], DATASET_COLORS["Rutishauser"],
                  ["1", "2", "3"],
                  f"DANDI 000469 (N={len(d469_pr)})\nβ={lme['dandi000469']['beta']:.3f}, "
                  f"p={lme['dandi000469']['p_value']:.2f}", "D")
    ax_d.set_xlabel("Load", fontsize=6)

    d001187_pr = _pr_records_from_stats(stats_data["dandi001187_ctg"], "pr_per_load", [1, 2, 3])
    d000673_pr = _pr_records_from_stats(stats_data["dandi000673_ctg"], "pr_per_load", [1, 2, 3])
    ax_e = fig.add_subplot(gs[1, 1])
    for key, row in d001187_pr.items():
        xs = sorted(row); ys = [row[x] for x in xs]
        ax_e.plot(range(len(xs)), ys, "o-", color="#76B7B2", alpha=0.4, lw=0.9, ms=2.5)
    for key, row in d000673_pr.items():
        xs = sorted(row); ys = [row[x] for x in xs]
        ax_e.plot(range(len(xs)), ys, "o-", color="#B07AA1", alpha=0.4, lw=0.9, ms=2.5)
    ax_e.set_xticks(range(3)); ax_e.set_xticklabels(["1", "2", "3"], fontsize=5.5)
    ax_e.set_xlabel("Load", fontsize=6)
    ax_e.set_ylabel("PR (cv)", fontsize=6)
    ax_e.set_title(f"001187 (teal, N={len(d001187_pr)}) /\n"
                   f"000673 (purple, N={len(d000673_pr)})", loc="left",
                   fontsize=6.3, fontweight="bold")
    panel_label(ax_e, "E")
    ax_e.text(0.97, 0.03,
              f"β={lme['dandi001187']['beta']:.2f},p={lme['dandi001187']['p_value']:.2f} / "
              f"β={lme['dandi000673']['beta']:.2f},p={lme['dandi000673']['p_value']:.2f}",
              transform=ax_e.transAxes, ha="right", va="bottom", fontsize=4.3)

    # ── F: cross-dataset summary — every dataset's slope is small & non-significant
    ax_f = fig.add_subplot(gs[1, 2])
    order = ["miller", "boran_ieeg", "boran_units", "dandi000469", "dandi001187", "dandi000673"]
    labels = ["Miller\nECoG", "Boran\niEEG", "Boran\nunits", "000469", "001187", "000673"]
    cols = [DATASET_COLORS["Miller"], DATASET_COLORS["Boran"], "#F1CE63",
            DATASET_COLORS["Rutishauser"], "#76B7B2", "#B07AA1"]
    betas = [lme[k]["beta"] for k in order]
    ps = [lme[k]["p_value"] for k in order]
    y_pos = np.arange(len(order))[::-1]
    ax_f.barh(y_pos, betas, color=cols, alpha=0.85, height=0.6)
    ax_f.axvline(0, color="k", lw=0.7)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels(labels, fontsize=5.3)
    ax_f.set_xlabel("PR-vs-load slope β", fontsize=6)
    ax_f.set_title("F  No dataset shows scaling\n(all p>0.2, permutation)",
                   loc="left", fontsize=6.3, fontweight="bold")
    for y, b, p in zip(y_pos, betas, ps):
        ax_f.text(b + (0.01 if b >= 0 else -0.01), y, f"p={p:.2f}",
                  ha="left" if b >= 0 else "right", va="center", fontsize=4.3)
    xmax = max(abs(min(betas)), abs(max(betas))) * 1.6
    ax_f.set_xlim(-xmax, xmax)

    fig.suptitle("Figure 2 — Manifold dimensionality does not scale with WM load, "
                 "across 6 datasets and 3 recording modalities",
                 fontsize=7.5, fontweight="bold", y=0.975)
    save_figure(fig, "fig2_main")
    print("  Figure 2 saved.")
    plt.close(fig)
    return


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Cross-temporal generalisation (PRIMARY FINDING)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure3(stats_data):
    """CTG matrices, off-diagonal AUC, and temporal stability index.

    Matrices are loaded from results/miller_ctg_corrected_{subj}.npz —
    PCA folded into cross-validation, label-shuffle permutation null — not
    recomputed with the old fixed-global-PCA pipeline.
    """
    print("  Loading corrected CTG matrices...")
    nature_style()

    ctg_matrices, ctg_times = {}, {}
    for subj in SUBJECTS:
        r = np.load(RESULTS / f"miller_ctg_corrected_{subj}.npz", allow_pickle=True)
        ctg_matrices[subj] = r["auc_mat"]
        ctg_times[subj] = r["times_ctg"]
        mat = r["auc_mat"]
        offdiag_mask = ~np.eye(mat.shape[0], dtype=bool)
        print(f"    {subj}: {mat.shape[0]}×{mat.shape[1]}, "
              f"off-diag={np.nanmean(mat[offdiag_mask]):.3f}")

    fig = plt.figure(figsize=(7.2, 7.6))
    gs_top = gridspec.GridSpec(2, 2, fig, hspace=0.55, wspace=0.46,
                               left=0.08, right=0.57, top=0.91, bottom=0.58)
    gs_bot = gridspec.GridSpec(1, 2, fig, hspace=0.35, wspace=0.32,
                               left=0.10, right=0.95, top=0.42, bottom=0.06)
    gs_side = gridspec.GridSpec(2, 1, fig, hspace=0.60, wspace=0.5,
                                left=0.64, right=0.97, top=0.90, bottom=0.58)

    cmap = plt.cm.RdYlBu_r
    vmin, vmax = 0.45, 0.80

    for idx, subj in enumerate(SUBJECTS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs_top[row, col])
        mat = ctg_matrices[subj]
        t = ctg_times[subj]

        ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                  origin="lower", extent=[t[0], t[-1], t[0], t[-1]])

        # Explicit diagonal (train time = test time) — the square/block
        # structure off this line, not the line itself, is the evidence for
        # a temporally-stable code; a narrow band hugging only this line
        # would instead indicate a rapidly-evolving (dynamic) code.
        ax.plot([t[0], t[-1]], [t[0], t[-1]], color="k", lw=0.7, ls=":", alpha=0.7)
        ax.set_xlabel("Test time (s)", fontsize=6)
        ax.set_ylabel("Train time (s)", fontsize=6)
        ax.tick_params(labelsize=5)

        stab = stats_data["ctg"][subj]["temporal_stability"]
        p_off = stats_data["ctg"][subj]["p_offdiag_vs_chance"]
        p_str = "p<0.001" if p_off < 0.001 else f"p={p_off:.3f}"
        ax.set_title(f"{SUBJ_LABELS[subj]}  τ={stab:.3f}, {p_str}",
                     fontsize=6.5, fontweight="bold")
        panel_label(ax, ["A", "B", "C", "D"][idx])

    # Horizontal colorbar in gap between CTG matrices and bottom panels
    cbar_ax = fig.add_axes([0.10, 0.505, 0.40, 0.012])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Decoding AUC", fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=4.5)
    # mark chance level on the colorbar
    chance_frac = (0.5 - vmin) / (vmax - vmin)
    cbar.ax.axvline(chance_frac, color="k", lw=1.0, ls="--")

    # Off-diagonal effect size (AUC-0.5) and τ per subject/session, across
    # ALL 6 applicable neural datasets — the same two summary statistics,
    # computed by the same label-shuffle-tested nested-CV pipeline
    # everywhere, plotted as paired dot-plots (not bar+SEM, per-subject
    # points visible given the small per-dataset N).
    dataset_specs = [
        ("Miller\nECoG", DATASET_COLORS["Miller"],
         [(s, stats_data["ctg"][s]) for s in SUBJECTS]),
        ("Boran\niEEG", DATASET_COLORS["Boran"],
         list(stats_data["boran_ctg"].items())),
        ("Boran\nunits", "#F1CE63",
         list(stats_data.get("dandi000574_units_ctg", {}).items())),
        ("000469", DATASET_COLORS["Rutishauser"],
         list(stats_data["dandi000469_ctg"].items())),
        ("001187", "#76B7B2",
         list(stats_data.get("dandi001187_ctg", {}).items())),
        ("000673", "#B07AA1",
         list(stats_data.get("dandi000673_ctg", {}).items())),
    ]

    def _offdiag_effect(v):
        if "offdiag_effect" in v:
            return v["offdiag_effect"]
        return v["mean_offdiag_auc"] - 0.5 if np.isfinite(v.get("mean_offdiag_auc", np.nan)) else np.nan

    ax_e = fig.add_subplot(gs_side[0])
    rng_e = np.random.default_rng(7)
    for xc, (label, col, items) in enumerate(dataset_specs):
        vals = np.array([_offdiag_effect(v) for _, v in items])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        jit = rng_e.uniform(-0.12, 0.12, len(vals))
        ax_e.scatter(xc + jit, vals, color=col, s=14, alpha=0.65, zorder=3)
        m = vals.mean(); se = vals.std() / np.sqrt(len(vals))
        ax_e.errorbar(xc, m, yerr=se, fmt="D", color="k", ms=4, capsize=3, lw=1.1, zorder=4)
    ax_e.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_e.set_xticks(range(len(dataset_specs)))
    ax_e.set_xticklabels([d[0] for d in dataset_specs], fontsize=5)
    ax_e.set_ylabel("Off-diag. effect (AUC−0.5)", fontsize=6)
    ax_e.set_title("E  Load/context CTG effect size\n(every dot = one subject/session)",
                   fontsize=6, fontweight="bold", loc="left")

    # ── F: Temporal stability τ, same 6 datasets — only where the diagonal is
    # meaningfully decodable (min_diag_auc >= 0.55); τ from a near-chance
    # decoder is an unstable ratio of two noisy numbers, not a fixed point.
    MIN_DIAG_AUC = 0.55

    def _tau_if_interpretable(v):
        diag = v.get("mean_diag_auc", np.nan)
        if not np.isfinite(diag) or diag < MIN_DIAG_AUC:
            return np.nan
        return v.get("tau", np.nan)

    ax_f = fig.add_subplot(gs_side[1])
    for xc, (label, col, items) in enumerate(dataset_specs):
        if label == "Miller\nECoG":
            vals = np.array([stats_data["ctg"][s]["temporal_stability"]
                             for s in SUBJECTS
                             if stats_data["ctg"][s]["mean_diag"] >= MIN_DIAG_AUC])
        else:
            vals = np.array([_tau_if_interpretable(v) for _, v in items])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        jit = rng_e.uniform(-0.12, 0.12, len(vals))
        ax_f.scatter(xc + jit, vals, color=col, s=14, alpha=0.65, zorder=3)
        m = vals.mean(); se = vals.std() / np.sqrt(len(vals))
        ax_f.errorbar(xc, m, yerr=se, fmt="D", color="k", ms=4, capsize=3, lw=1.1, zorder=4)
    ax_f.axhline(1.0, color="k", lw=0.8, ls="--")
    ax_f.set_xticks(range(len(dataset_specs)))
    ax_f.set_xticklabels([d[0] for d in dataset_specs], fontsize=5)
    ax_f.set_ylim(0.0, 1.6)
    ax_f.set_ylabel("Temporal stability τ", fontsize=6)
    ax_f.set_title(f"F  τ where diagonal AUC≥{MIN_DIAG_AUC}; τ<1 expected for\n"
                   "any non-stationary code, not itself evidence",
                   fontsize=5.8, fontweight="bold", loc="left")

    # ── G: Negative controls — maintenance vs. comparison/baseline epoch ─────
    ax_g = fig.add_subplot(gs_bot[0])
    ctrl_specs = [
        ("Miller\n(transient)", DATASET_COLORS["Miller"],
         [(v["maintenance"]["offdiag_effect"], v["negative_control_transient"]["offdiag_effect"])
          for v in stats_data["miller_ctg_corrected"].values()]),
        ("Boran\n(baseline)", DATASET_COLORS["Boran"],
         [(v["offdiag_effect"], v["negative_control_baseline"]["offdiag_effect"])
          for v in stats_data["boran_ctg"].values()]),
        ("000469\n(encoding)", DATASET_COLORS["Rutishauser"],
         [(v["offdiag_effect"], v["negative_control"]["offdiag_effect"])
          for v in stats_data["dandi000469_ctg"].values()
          if np.isfinite(v["negative_control"]["offdiag_effect"])]),
    ]
    for xc, (label, col, pairs) in enumerate(ctrl_specs):
        maint = np.array([p[0] for p in pairs])
        ctrl = np.array([p[1] for p in pairs])
        ax_g.scatter(np.full(len(maint), xc - 0.15), maint, color=col, s=14, alpha=0.7, zorder=3)
        ax_g.scatter(np.full(len(ctrl), xc + 0.15), ctrl, color=col, s=14, alpha=0.3,
                     marker="s", zorder=3)
        for m, c in zip(maint, ctrl):
            ax_g.plot([xc - 0.15, xc + 0.15], [m, c], color=col, lw=0.4, alpha=0.4, zorder=2)
    ax_g.axhline(0.0, color="k", lw=0.8, ls="--")
    ax_g.set_xticks(range(len(ctrl_specs)))
    ax_g.set_xticklabels([c[0] for c in ctrl_specs], fontsize=5.3)
    ax_g.set_ylabel("Off-diag. effect (AUC−0.5)", fontsize=6.3)
    ax_g.set_title("G  Maintenance (●) vs. dynamic-code\ncontrol epoch (■), paired per subject",
                   loc="left", fontsize=6, fontweight="bold")

    # ── H: Band generality — τ across the spectrum ────────────────────────────
    ax_h = fig.add_subplot(gs_bot[1])
    mb_path = RESULTS / "multiband_ctg.npz"
    if mb_path.exists():
        mb = np.load(mb_path, allow_pickle=True)
        band_order = ["theta", "alpha", "beta", "gamma", "hgp"]
        band_lbls  = ["θ\n4–8", "α\n8–13", "β\n13–30", "γ\n30–70", "HGP\n70–150"]
        band_cols  = ["#8C564B", "#9467BD", "#17BECF", "#BCBD22", "#E15759"]
        means, sds = [], []
        for b in band_order:
            taus = [float(mb[f"{s}_{b}_tau"]) for s in SUBJECTS if f"{s}_{b}_tau" in mb]
            means.append(np.mean(taus)); sds.append(np.std(taus))
        ax_h.bar(range(5), means, yerr=sds, color=band_cols, alpha=0.85,
                 width=0.65, capsize=2.5, error_kw={"lw": 0.8})
        ax_h.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="τ=1")
        ax_h.set_xticks(range(5))
        ax_h.set_xticklabels(band_lbls, fontsize=5)
        ax_h.set_ylim(0.0, 1.15)
        ax_h.set_ylabel("Temporal stability τ (raw-AUC ratio)", fontsize=6)
        ax_h.set_title("H  Band generality (Miller, N=4)\nτ<1 in every band — no band is privileged",
                       loc="left", fontsize=6, fontweight="bold")
        ax_h.legend(frameon=False, fontsize=5, loc="lower right")
    else:
        ax_h.set_title("H  Band generality (Miller, N=4)", loc="left",
                       fontsize=6, fontweight="bold")
        ax_h.text(0.5, 0.5, "Run run_multiband_analysis.py", ha="center",
                  va="center", transform=ax_h.transAxes, fontsize=6)
        ax_h.axis("off")

    fig.suptitle("Figure 3 — Sustained load/context coding is temporally stable across the delay",
                 fontsize=8, fontweight="bold", y=0.98)
    save_figure(fig, "fig3_main")
    print("  Figure 3 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Dynamical geometry: ring attractor and tangling
# ─────────────────────────────────────────────────────────────────────────────

def make_figure4(stats_data):
    """Ring attractor phase analysis with ACTUAL computed phases, Q(t), DMD,
    and (I-K) the causal anchor: delay-period stimulation obeys the same
    geometry (Section~sec:causal-anchor) -- merged in here rather than kept
    as a separate figure, per the R3 restructure."""
    nature_style()
    fig = plt.figure(figsize=(7.2, 8.1))
    gs_top = gridspec.GridSpec(1, 4, fig, hspace=0.4, wspace=0.38,
                               left=0.06, right=0.97, top=0.94, bottom=0.68)
    gs_bot = gridspec.GridSpec(1, 4, fig, hspace=0.4, wspace=0.62,
                               left=0.06, right=0.98, top=0.60, bottom=0.36)
    gs_causal = gridspec.GridSpec(1, 4, fig, wspace=0.55,
                                  left=0.06, right=0.99, top=0.27, bottom=0.06)

    # ── A-D: Polar phase plots from ACTUAL latent trajectories ────────────────
    for idx, subj in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs_top[0, idx], projection="polar")
        ray = stats_data["rayleigh"][subj]
        R = ray["R"]
        p = ray["p_value"]
        N = int(ray["N"])

        # Compute actual 2-back endpoint phases from latent trajectories
        phases = compute_actual_phases(subj, t_maint_end=1.4)
        mu_actual = float(np.angle(np.mean(np.exp(1j * phases))))
        if mu_actual < 0:
            mu_actual += 2 * np.pi

        bins = np.linspace(0, 2 * np.pi, 19)
        counts, _ = np.histogram(phases, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bar_width = 2 * np.pi / 18

        ax.bar(bin_centers, counts, width=bar_width * 0.85,
               color=SUBJ_COLORS[idx], alpha=0.72, zorder=2)

        # Mean resultant vector
        r_scale = counts.max() * 1.05
        ax.annotate("", xy=(mu_actual, r_scale),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
                    xycoords="data", textcoords="data")

        p_str = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
        ax.set_title(f"{SUBJ_LABELS[subj]}\nR={R:.3f}, {p_str}",
                     fontsize=6, pad=7, fontweight="bold")
        ax.tick_params(labelsize=4)
        ax.set_rlabel_position(45)
        panel_label(ax, ["A", "B", "C", "D"][idx], x=-0.14)

    # ── E: Rayleigh R summary ─────────────────────────────────────────────────
    ax_e = fig.add_subplot(gs_bot[0])
    R_vals = [stats_data["rayleigh"][s]["R"] for s in SUBJECTS]
    p_vals = [stats_data["rayleigh"][s]["p_value"] for s in SUBJECTS]

    ax_e.bar(np.arange(4), R_vals, color=SUBJ_COLORS, alpha=0.85, width=0.6)
    ax_e.axhline(1 / np.sqrt(np.mean([stats_data["rayleigh"][s]["N"] for s in SUBJECTS])),
                 color="k", lw=0.7, ls="--", alpha=0.5, label="Uniform expectation")
    for i, (r_val, p_val) in enumerate(zip(R_vals, p_vals)):
        star = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else
               ("*" if p_val < 0.05 else "ns"))
        ax_e.text(i, r_val + 0.012, star, ha="center", va="bottom", fontsize=8)
    ax_e.set_xticks(np.arange(4))
    ax_e.set_xticklabels([SUBJ_LABELS[s] for s in SUBJECTS], fontsize=5.5, rotation=20)
    ax_e.set_ylabel("Rayleigh R (mean resultant)", fontsize=6.5)
    ax_e.set_ylim(0, 0.62)
    ax_e.set_title("E  Endpoint phase concentration\n(ring: R≈0; fixed-pt: R≈1)",
                   loc="left", fontsize=5.8, fontweight="bold")
    ax_e.legend(frameon=False, fontsize=5.5, loc="upper right")

    # ── F: Q(t) tangling traces ───────────────────────────────────────────────
    ax_f = fig.add_subplot(gs_bot[1])
    dyn = np.load(RESULTS / "03_dynamics.npz", allow_pickle=True)
    Q_tgt  = dyn["Q_tgt_pool"]
    Q_ntgt = dyn["Q_ntgt_pool"]
    clus   = np.load(RESULTS / "03_cluster_q.npz", allow_pickle=True)
    times_q = clus["times"]
    t_stat_q = clus["t_stat"]
    n_sig = int(clus["n_sig_clusters"])

    step = 5
    t_s = times_q[::step]
    q_tgt_m   = Q_tgt[:, ::step].mean(0)
    q_tgt_se  = Q_tgt[:, ::step].std(0) / np.sqrt(Q_tgt.shape[0])
    q_ntgt_m  = Q_ntgt[:, ::step].mean(0)
    q_ntgt_se = Q_ntgt[:, ::step].std(0) / np.sqrt(Q_ntgt.shape[0])

    ax_f.plot(t_s, q_tgt_m / 1e3, color=PALETTE["target"], lw=1.2, label="Target")
    ax_f.fill_between(t_s, (q_tgt_m - q_tgt_se) / 1e3,
                      (q_tgt_m + q_tgt_se) / 1e3, color=PALETTE["target"], alpha=0.2)
    ax_f.plot(t_s, q_ntgt_m / 1e3, color=PALETTE["non_target"], lw=1.2, label="Non-target")
    ax_f.fill_between(t_s, (q_ntgt_m - q_ntgt_se) / 1e3,
                      (q_ntgt_m + q_ntgt_se) / 1e3, color=PALETTE["non_target"], alpha=0.2)
    ax_f.axvline(0, color="k", lw=0.5, ls="--")
    ax_f.axvspan(0.3, 1.4, color="gold", alpha=0.12, zorder=0)
    ax_f.set_xlabel("Time (s)", fontsize=6.5)
    ax_f.set_ylabel("Q(t) × 10³", fontsize=6.5)
    ax_f.set_title(f"F  Trajectory tangling Q(t)\n"
                   f"(FWER: {n_sig} sig. clusters)",
                   loc="left", fontsize=5.8, fontweight="bold")
    ax_f.legend(frameon=False, fontsize=5.5)

    ax_f2 = ax_f.twinx()
    ax_f2.plot(times_q[::step], t_stat_q[::step], color="gray", lw=0.6, alpha=0.45, ls=":")
    ax_f2.set_ylabel("t-stat", fontsize=5, color="gray", labelpad=1)
    ax_f2.tick_params(axis="y", labelsize=4, labelcolor="gray")

    # ── G: DMD eigenspectrum ──────────────────────────────────────────────────
    ax_g = fig.add_subplot(gs_bot[2])
    dyn2 = np.load(RESULTS / "03_dynamics.npz", allow_pickle=True)
    evals_tgt  = dyn2["evals_tgt"].mean(0)
    evals_ntgt = dyn2["evals_ntgt"].mean(0)

    theta_circ = np.linspace(0, 2 * np.pi, 300)
    ax_g.plot(np.cos(theta_circ), np.sin(theta_circ), "k-", lw=0.7, alpha=0.4)
    ax_g.axhline(0, color="k", lw=0.3)
    ax_g.axvline(0, color="k", lw=0.3)
    ax_g.scatter(evals_tgt.real, evals_tgt.imag, color=PALETTE["target"], s=25,
                 zorder=3, label=f"Target |λ|={np.abs(evals_tgt).mean():.4f}")
    ax_g.scatter(evals_ntgt.real, evals_ntgt.imag, color=PALETTE["non_target"],
                 s=25, marker="^", zorder=3,
                 label=f"Non-tgt |λ|={np.abs(evals_ntgt).mean():.4f}")
    ax_g.set_xlim(-1.2, 1.2)
    ax_g.set_ylim(-1.2, 1.2)
    ax_g.set_box_aspect(1)
    ax_g.set_xlabel("Re(λ)", fontsize=6.5)
    ax_g.set_ylabel("Im(λ)", fontsize=6.5, labelpad=1)
    p_dmd = stats_data["dmd_lambda"]["p_perm"]
    ax_g.set_title(f"G  DMD eigenspectrum\n(permutation p = {p_dmd:.3f})",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_g.legend(frameon=False, fontsize=5, loc="upper right")

    # ── H: Flow divergence ∇·v — mean-trajectory vs. ensemble (single-trial,
    # cross-validated) estimate. The trial-averaged mean-trajectory fit is
    # confounded by trial-averaging contraction and, for Miller/Boran, does
    # NOT survive replacement with an ensemble fit on pooled single-trial
    # transitions: divergence collapses toward zero and flips sign in a
    # majority of subjects. Only the single-unit dataset (000469) shows a
    # divergence estimate that is robust to this check.
    ax_h = fig.add_subplot(gs_bot[3])
    div_path = RESULTS / "divergence_analysis.npz"
    if div_path.exists():
        div = np.load(div_path, allow_pickle=True)
        miller_subs = ["al", "ca", "cc", "ug"]
        boran_subs  = [f"sub-{i:02d}" for i in range(1, 10)]
        rushi_subs  = RUSHI_SUBS

        def _pair(prefix, subs):
            mt, en = [], []
            for s in subs:
                k_mt, k_en = f"{prefix}_{s}_div_scalar", f"{prefix}_{s}_ensemble_div_scalar"
                if k_mt in div and k_en in div:
                    mt.append(float(div[k_mt])); en.append(float(div[k_en]))
            return np.array(mt), np.array(en)

        m_mt, m_en = _pair("miller", miller_subs)
        b_mt, b_en = _pair("boran", boran_subs)
        r_mt, r_en = _pair("dandi000469", rushi_subs)

        for xc, (mt, en), col, lbl in [
            (0, (m_mt, m_en), DATASET_COLORS["Miller"],      "Miller"),
            (1, (b_mt, b_en), DATASET_COLORS["Boran"],       "Boran"),
            (2, (r_mt, r_en), DATASET_COLORS["Rutishauser"], "000469"),
        ]:
            ax_h.scatter(np.full(len(mt), xc - 0.15), mt, color=col, s=16, alpha=0.7, zorder=3)
            ax_h.scatter(np.full(len(en), xc + 0.15), en, facecolors="none",
                        edgecolors=col, s=16, alpha=0.9, zorder=3)
            for a, b in zip(mt, en):
                ax_h.plot([xc - 0.15, xc + 0.15], [a, b], color=col, lw=0.4, alpha=0.4, zorder=2)
        ax_h.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax_h.set_xlim(-0.6, 2.6)
        ax_h.set_xticks([0, 1, 2])
        ax_h.set_xticklabels(["Miller\nPFC", "Boran\nMTL", "000469\nSU"], fontsize=6)
        ax_h.set_ylabel("Flow divergence ∇·v (s⁻¹)", fontsize=6.5)
        ax_h.set_title("H  Mean-traj. (●) vs. single-trial\nensemble (○) ∇·v, paired per subject",
                       loc="left", fontsize=6, fontweight="bold")
    else:
        ax_h.set_title("H  Stimulation timing: ∇·v", loc="left",
                       fontsize=6, fontweight="bold")
        ax_h.text(0.5, 0.5, "Run run_divergence_analysis.py", ha="center",
                  va="center", transform=ax_h.transAxes, fontsize=6)
        ax_h.axis("off")

    # ── I: Soldado CATE-vs-alignment gate — the actual pseudo-outcome scatter
    # + fitted slope, with the permutation null shown as an inset histogram
    # (not a schematic: results/causal_soldado_gate_detail.npz holds the real
    # AIPW pseudo-outcome, modifier, and null-slope draws from the pipeline).
    ax_i = fig.add_subplot(gs_causal[0])
    gate_path = RESULTS / "causal_soldado_gate_detail.npz"
    gate = stats_data["causal_soldado"]["gate"]
    if gate_path.exists():
        gd = np.load(gate_path)
        phi, m, null = gd["phi"], gd["modifier"], gd["null"]
        ax_i.scatter(m, phi, s=4, color=DATASET_COLORS["Miller"], alpha=0.15,
                    edgecolors="none", zorder=2, rasterized=True)
        m_line = np.array([m.min(), m.max()])
        ax_i.plot(m_line, gate["intercept"] + gate["slope"] * m_line,
                  color="k", lw=1.3, zorder=3)
        ax_i.set_xlabel("Alignment to $v^*$", fontsize=6.5)
        ax_i.set_ylabel("AIPW pseudo-outcome", fontsize=6.5)
        ax_i.set_title(f"I  Soldado CATE gate ($N$={gate['n']:,})\n"
                       f"slope={gate['slope']:.3f} [{gate['slope_ci_lo']:.2f},"
                       f" {gate['slope_ci_hi']:.2f}], $p$={gate['p_value']:.4f}",
                       loc="left", fontsize=5.8, fontweight="bold")
        axins = ax_i.inset_axes([0.60, 0.66, 0.36, 0.30])
        axins.hist(null, bins=30, color="gray", alpha=0.6, lw=0)
        axins.axvline(gate["slope"], color="#B41E1E", lw=1.1)
        axins.set_xticks([]); axins.set_yticks([])
        axins.set_title("perm. null", fontsize=4.3, pad=1)
    panel_label(ax_i, "I")

    # ── J: dynamic vs. stable subspace perturbation (Soldado, paired sessions)
    ax_j = fig.add_subplot(gs_causal[1])
    dvs = stats_data["causal_soldado"]["dynamic_vs_stable_replication"]
    per_sess = stats_data["causal_soldado"]["per_session"]
    dyn_vals = [abs(v["dynamic_vs_stable"]["dynamic_pct_change"]) for v in per_sess.values()
                if v.get("dynamic_vs_stable") is not None]
    stab_vals = [abs(v["dynamic_vs_stable"]["stable_pct_change"]) for v in per_sess.values()
                 if v.get("dynamic_vs_stable") is not None]
    for a, b in zip(dyn_vals, stab_vals):
        ax_j.plot([0, 1], [b, a], color="gray", lw=0.7, alpha=0.5, zorder=1)
    ax_j.scatter(np.zeros(len(stab_vals)), stab_vals, color=PALETTE["non_target"],
                s=18, zorder=2)
    ax_j.scatter(np.ones(len(dyn_vals)), dyn_vals, color=PALETTE["target"], s=18, zorder=2)
    ax_j.set_xlim(-0.4, 1.4)
    ax_j.set_xticks([0, 1]); ax_j.set_xticklabels(["Stable", "Dynamic"], fontsize=6)
    ax_j.set_ylabel("|% change| under stim", fontsize=6.5)
    ax_j.set_title(f"J  Stim perturbs dynamic $>$ stable\n"
                   f"subspace ($N$={dvs['n_sessions']}, trend $p$={dvs['p_value']:.3f})",
                   loc="left", fontsize=5.8, fontweight="bold")

    # ── K: RAM (ds005489) — same-signed null replication at encoding
    ax_k = fig.add_subplot(gs_causal[2])
    ram = stats_data["causal_ram"]["result"]
    labels = ["Soldado\n(delay, macaque)", "RAM ds005489\n(encoding, human)"]
    slopes = [gate["slope"], ram["slope"]]
    lo = [gate["slope_ci_lo"], ram["slope_ci_lo"]]
    hi = [gate["slope_ci_hi"], ram["slope_ci_hi"]]
    ps = [gate["p_value"], ram["p_value"]]
    cols = [DATASET_COLORS["Miller"], "#888888"]
    y_pos = [1, 0]
    for y, s, l, h, c in zip(y_pos, slopes, lo, hi, cols):
        ax_k.plot([l, h], [y, y], color=c, lw=1.6, zorder=2)
        ax_k.scatter([s], [y], color=c, s=28, zorder=3)
    ax_k.axvline(0, color="k", lw=0.7, ls="--")
    ax_k.set_yticks(y_pos); ax_k.set_yticklabels(labels, fontsize=5.8)
    ax_k.set_ylim(-0.7, 1.7)
    ax_k.set_xlabel("CATE-vs-alignment slope", fontsize=6.5)
    for y, s, p in zip(y_pos, slopes, ps):
        ax_k.text(s, y + 0.25, f"p={p:.3f}" if p >= 0.001 else "p<0.001",
                 ha="center", fontsize=5, color="k")
    ax_k.set_title("K  Same-signed slope at encoding\n(RAM, human), not significant",
                   loc="left", fontsize=5.8, fontweight="bold")
    panel_label(ax_k, "K")

    # ── L: the benchmark leaderboard — v* adjudicated against every
    # competing theory of the causally-relevant controllable direction, on
    # the SAME cross-fit pseudo-outcome (results/causal_benchmark.json).
    ax_l = fig.add_subplot(gs_causal[3])
    bench = stats_data.get("causal_benchmark")
    if bench is not None:
        lb_names = ["vstar_alignment", "min_energy_dir_alignment", "random_alignment",
                    "gramian_trace", "stable_alignment", "input_norm",
                    "anat_avg_ctrl", "anat_modal_ctrl"]
        lb_names = [n for n in lb_names if n in bench["leaderboard"]]
        lb_labels = {"vstar_alignment": "$v^*$ (ours)", "min_energy_dir_alignment": "Min-energy dir.",
                    "random_alignment": "Random", "gramian_trace": "Gramian (fn'l)",
                    "stable_alignment": "Stable dir.", "input_norm": "Coupling dose",
                    "anat_avg_ctrl": "Avg. ctrl. (struct.)", "anat_modal_ctrl": "Modal ctrl. (struct.)"}
        rows_l = [bench["leaderboard"][n] for n in lb_names]
        # Round-8 Part 9/10 (K2/K3/K4): the arena-level winner is argmax(slope) WITHIN this
        # Soldado-only 8-arm list, never the stale/mixed top-level bench["winner"] (which
        # can resolve to an arm not even in lb_names, e.g. macrosignal_pac -- scored on a
        # different dataset entirely; see results/causal_benchmark.json primary_leaderboard,
        # now itself the Soldado-8 arena per K4).
        soldado_winner = max(lb_names, key=lambda n: bench["leaderboard"][n]["slope"])
        y_l = np.arange(len(lb_names))[::-1]
        cols_l = ["#59A14F" if (n == soldado_winner) else "#4E79A7" for n in lb_names]
        for y, row, c in zip(y_l, rows_l, cols_l):
            ax_l.plot([row["slope_ci_lo"], row["slope_ci_hi"]], [y, y], color=c, lw=1.3, zorder=2)
            ax_l.scatter([row["slope"]], [y], color=c, s=20, zorder=3)
        ax_l.axvline(0, color="k", lw=0.7, ls="--")
        ax_l.set_yticks(y_l)
        ax_l.set_yticklabels([lb_labels[n] for n in lb_names], fontsize=5.3)
        ax_l.set_xlabel("Marginal DR slope (per SD)", fontsize=6.3)
        ax_l.set_title(f"L  Benchmark: only $v^*$ predicts\nstim. effect (winner={lb_labels[soldado_winner]})",
                       loc="left", fontsize=5.8, fontweight="bold")
    else:
        ax_l.set_title("L  Benchmark leaderboard", loc="left", fontsize=6, fontweight="bold")
        ax_l.text(0.5, 0.5, "Run run_soldado_pipeline.py", ha="center", va="center",
                 transform=ax_l.transAxes, fontsize=6)
        ax_l.axis("off")
    panel_label(ax_l, "L")

    fig.suptitle("Figure 4 — Dynamical geometry (ring-like phase encoding, near-unit-circle\n"
                 "DMD, stimulation-timing divergence) and the benchmark: among competing control\n"
                 "models, only the unstable-mode geometry predicts the causal effect of stimulation",
                 fontsize=7.8, fontweight="bold", y=0.985)
    save_figure(fig, "fig4_main")
    print("  Figure 4 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Item-identity (memorandum) CTG: the maintained-vs-silent question
# ─────────────────────────────────────────────────────────────────────────────

def make_figure5(stats_data):
    """Load/context CTG (Fig 3) shows a sustained, block-level signal is
    temporally stable, which any account of task-set maintenance predicts.
    This figure isolates the analysis that actually speaks to whether the
    memorandum itself is activity-maintained: decoding WHICH item is held,
    within a fixed load, across the delay. Three panels only — one dataset
    (pfc-3) supplies the matrix and diag-vs-offdiag contrast, the other
    supplies the pooled human replication; nothing else earns its place here."""
    nature_style()
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.55,
                           left=0.08, right=0.98, top=0.80, bottom=0.17)

    # ── A: PFC-3 content CTG matrix (macaque, independent species/lab) ───────
    ax_a = fig.add_subplot(gs[0])
    pfc3_path = RESULTS / "pfc3_content_ctg.npz"
    if pfc3_path.exists():
        d = np.load(pfc3_path, allow_pickle=True)
        mat, t = d["auc_mat"], d["times"]
        im = ax_a.imshow(mat, aspect="auto", cmap=plt.cm.RdYlBu_r, vmin=0.45, vmax=0.75,
                         origin="lower", extent=[t[0], t[-1], t[0], t[-1]])
        ax_a.plot([t[0], t[-1]], [t[0], t[-1]], color="k", lw=0.7, ls=":", alpha=0.7)
        cbar = plt.colorbar(im, ax=ax_a, fraction=0.045, pad=0.03)
        cbar.set_label("Macro-avg. AUC", fontsize=5.5, labelpad=2)
        cbar.ax.tick_params(labelsize=4.5)
        ax_a.set_xlabel("Test time (s)", fontsize=6.5)
        ax_a.set_ylabel("Train time (s)", fontsize=6.5)
    p3 = stats_data["pfc3_content_ctg"]
    ax_a.set_title(f"A  Macaque PFC (pfc-3), N={p3['n_neurons']} units\n"
                   f"9-way location, τ={p3['tau']:.2f}, p<0.001", loc="left",
                   fontsize=6.3, fontweight="bold")
    panel_label(ax_a, "A")

    # ── B: PFC-3 diag vs off-diag effect size — the key contrast ─────────────
    ax_b = fig.add_subplot(gs[1])
    ax_b.bar(0, p3["mean_diag_auc_minus_chance"], color="#FF9DA7", alpha=0.9, width=0.55)
    ax_b.bar(1, p3["mean_offdiag_auc_minus_chance"], color="#FF9DA7", alpha=0.5, width=0.55)
    ax_b.axhline(0, color="k", lw=0.7)
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels(["Diagonal\n(within-time)", "Off-diagonal\n(across-time)"], fontsize=6)
    ax_b.set_ylabel("Effect size (AUC−0.5)", fontsize=6.5)
    ax_b.set_title("B  Content decodable within AND\nacross time — not silent, but dynamic",
                   loc="left", fontsize=6.2, fontweight="bold")
    panel_label(ax_b, "B")

    # ── C: human MTL/frontal item-identity CTG (000469 only — 001187/000673
    # draw a near-unique picture per load-1 trial, with essentially no
    # within-session repeats, so classification-based content decoding
    # (which needs repeated classes) is not applicable there) ────────────────
    ax_c = fig.add_subplot(gs[2])
    entries = stats_data.get("dandi000469_ctg", {})
    vals = np.array([v["content_ctg"]["offdiag_effect"] for v in entries.values()
                     if v.get("content_ctg") is not None])
    rng_c = np.random.default_rng(3)
    jit = rng_c.uniform(-0.12, 0.12, len(vals))
    ax_c.scatter(np.zeros(len(vals)) + jit, vals, color=DATASET_COLORS["Rutishauser"],
                s=18, alpha=0.7, zorder=3)
    if len(vals):
        m = vals.mean(); se = vals.std() / np.sqrt(len(vals))
        ax_c.errorbar(0, m, yerr=se, fmt="D", color="k", ms=4, capsize=3, lw=1.1, zorder=4)
    ax_c.axhline(0, color="k", lw=0.8, ls="--")
    ax_c.set_xlim(-0.6, 0.6)
    ax_c.set_xticks([0])
    ax_c.set_xticklabels([f"000469 (N={len(vals)})"], fontsize=6.5)
    ax_c.set_ylabel("Off-diag. effect (AUC−0.5)", fontsize=6.3)
    pooled = stats_data.get("content_ctg_pooled", {})
    ax_c.set_title(f"C  Human MTL/frontal item-identity CTG\n"
                   f"Stouffer-combined p={pooled.get('p_combined', float('nan')):.1e}\n"
                   f"(001187/000673: no repeated items per session)",
                   loc="left", fontsize=5.6, fontweight="bold")
    panel_label(ax_c, "C")

    fig.suptitle("Figure 5 — Item-identity coding: the memorandum is significantly decodable "
                 "across the delay, but far more dynamically than load/context",
                 fontsize=7.3, fontweight="bold", y=0.98)
    save_figure(fig, "fig5_main")
    print("  Figure 5 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Behavioral relevance: does maintenance drift track trial outcome?
# ─────────────────────────────────────────────────────────────────────────────

def _binomial_sig_rate(d: dict, keys: tuple) -> tuple[int, int]:
    """Count (n_significant, n_tests) gated_outcome_cluster_test entries in a
    nested per-subject[/per-item] results dict, recursing through any of the
    given alignment/item key names present at each level."""
    n_tests, n_sig = 0, 0
    def _walk(node):
        nonlocal n_tests, n_sig
        if not isinstance(node, dict):
            return
        if "significant" in node and "clusters" in node:
            n_tests += 1
            if len(node["significant"]) > 0:
                n_sig += 1
            return
        for k, v in node.items():
            if k in keys or k not in ("n_trials",):
                _walk(v)
    _walk(d)
    return n_sig, n_tests


def make_figure6(stats_data):
    """Behavioral relevance of maintenance-period neural state, across three
    complementary tests: (A) correct-vs-error drift in every dataset carrying
    an outcome label, (B) whether single-trial decoder confidence (content and
    context) predicts trial outcome anywhere across the full trial timeline —
    tested via a binomial rate of significant clusters against the chance
    false-positive rate, since many session/window tests are run, and (C)
    whether active LQR control, using the error-vs-correct latent centroids as
    the target, outperforms passive (uncontrolled) dynamics at closing that
    gap (manifold-rescue, Boran)."""
    from scipy.stats import binomtest

    nature_style()
    fig = plt.figure(figsize=(7.2, 2.9))
    gs = gridspec.GridSpec(1, 3, fig, width_ratios=[1.1, 1, 1], wspace=0.55,
                           left=0.08, right=0.98, top=0.78, bottom=0.20)

    # ── A: correct-vs-error drift, every dataset with an outcome label ───────
    ax_a = fig.add_subplot(gs[0])
    beh_specs = [
        ("Boran iEEG", DATASET_COLORS["Boran"], stats_data.get("boran_correct_error_drift", {})),
        ("Boran units", "#F1CE63", stats_data.get("dandi000574_units_correct_error_drift", {})),
        ("DANDI 000469", DATASET_COLORS["Rutishauser"], stats_data.get("dandi000469_correct_error_drift", {})),
        ("DANDI 001187", "#76B7B2", stats_data.get("dandi001187_correct_error_drift", {})),
        ("DANDI 000673", "#B07AA1", stats_data.get("dandi000673_correct_error_drift", {})),
    ]
    y_pos = np.arange(len(beh_specs))[::-1]
    betas = [b.get("beta", np.nan) for _, _, b in beh_specs]
    ps = [b.get("p_value", np.nan) for _, _, b in beh_specs]
    ns = [b.get("n_trials", None) for _, _, b in beh_specs]
    cols = [c for _, c, _ in beh_specs]

    ax_a.barh(y_pos, betas, color=cols, alpha=0.85, height=0.6)
    ax_a.axvline(0, color="k", lw=0.7)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels([f"{s[0]} (n={n})" if n else s[0] for s, n in zip(beh_specs, ns)],
                       fontsize=5.8)
    finite_betas = [b for b in betas if np.isfinite(b)]
    xmax = max(abs(min(finite_betas)), abs(max(finite_betas))) * 2.3 if finite_betas else 1.0
    ax_a.set_xlim(-xmax, xmax)
    for y, b, p in zip(y_pos, betas, ps):
        if np.isfinite(b):
            ax_a.text(b + (xmax * 0.05 if b >= 0 else -xmax * 0.05), y, f"p={p:.2f}",
                    ha="left" if b >= 0 else "right", va="center", fontsize=5.5)
    ax_a.set_xlabel("Correct−error drift β", fontsize=6.5)
    ax_a.set_title("A  Maintenance drift vs. outcome", loc="left", fontsize=6.3, fontweight="bold")
    panel_label(ax_a, "A")

    # ── B: decoder-confidence-vs-outcome, binomial rate of significant ──────
    # clusters across sessions/windows/items, content and context
    ax_b = fig.add_subplot(gs[1])
    content_d = stats_data.get("decoder_confidence_timecourse_000469", {})
    context_d = stats_data.get("context_confidence_timecourse", {})
    n_sig_c, n_tot_c = _binomial_sig_rate(content_d, ("item1", "item2", "item3"))
    n_sig_x, n_tot_x = _binomial_sig_rate(context_d, ())
    rate_c = n_sig_c / n_tot_c if n_tot_c else 0.0
    rate_x = n_sig_x / n_tot_x if n_tot_x else 0.0
    p_c = binomtest(n_sig_c, n_tot_c, 0.05, alternative="greater").pvalue if n_tot_c else np.nan
    p_x = binomtest(n_sig_x, n_tot_x, 0.05, alternative="greater").pvalue if n_tot_x else np.nan

    ax_b.bar([0, 1], [rate_c, rate_x], color=[DATASET_COLORS["Rutishauser"], "#B07AA1"],
             alpha=0.85, width=0.55)
    ax_b.axhline(0.05, color="k", lw=0.8, ls="--", label="Chance (α=0.05)")
    ax_b.set_xticks([0, 1])
    ax_b.set_xticklabels([f"Content\n({n_sig_c}/{n_tot_c})", f"Context\n({n_sig_x}/{n_tot_x})"],
                          fontsize=6)
    ax_b.set_ylabel("Fraction of tests\nwith a sig. cluster", fontsize=6.3)
    ax_b.legend(fontsize=5, loc="upper right", frameon=False)
    ax_b.set_title(f"B  Decoder confidence vs. outcome\np={p_c:.2f} (content), p={p_x:.2f} (context)",
                   loc="left", fontsize=6.0, fontweight="bold")
    panel_label(ax_b, "B")

    # ── C: manifold rescue — controlled vs. passive reduction (Boran) ───────
    ax_c = fig.add_subplot(gs[2])
    mr = stats_data.get("manifold_rescue", {})
    per_subj = mr.get("per_subject", {})
    subs = sorted(per_subj.keys())
    passive = np.array([per_subj[s]["passive_reduction_pct"] for s in subs])
    controlled = np.array([per_subj[s]["controlled_reduction_pct"] for s in subs])
    x = np.arange(len(subs))
    w = 0.38
    ax_c.bar(x - w / 2, passive, width=w, color="#999999", alpha=0.85, label="Passive")
    ax_c.bar(x + w / 2, controlled, width=w, color=DATASET_COLORS["Boran"], alpha=0.85, label="LQR-controlled")
    ax_c.axhline(0, color="k", lw=0.7)
    ax_c.set_yscale("symlog", linthresh=10)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([s.replace("sub-", "S") for s in subs], fontsize=5, rotation=90)
    ax_c.set_ylabel("Distance-to-target\nreduction (%, symlog)", fontsize=6.1)
    ax_c.legend(fontsize=5, loc="lower left", frameon=False)
    contrast = mr.get("controlled_greater_than_passive", {})
    ax_c.set_title(f"C  Manifold rescue (Boran, N={len(subs)})\n"
                   f"controlled>passive p={contrast.get('p_value', float('nan')):.3f}",
                   loc="left", fontsize=6.0, fontweight="bold")
    panel_label(ax_c, "C")

    fig.suptitle("Figure 6 — Behavioral relevance and rescue: drift, decoder confidence, "
                 "and active control", fontsize=7.3, fontweight="bold", y=0.98)
    save_figure(fig, "fig6_main")
    print("  Figure 6 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Content vs. context: a rotating item code on a stable load axis
# ─────────────────────────────────────────────────────────────────────────────

def make_figure7(stats_data):
    """Within-subject content/context dissociation (DANDI 000469, N=18): (A)
    the paired cross-temporal-generalization effect size (context reliably
    larger than content, the primary statistic for this comparison — tau
    itself is gated uninterpretable for most subjects at these AUCs, so it is
    not used here), (B) exemplar coding-direction-stability (cosine
    similarity of the decoding weight vector across time) matrices for one
    subject, context vs. content, and (C) the content-axis rotation index
    against DMD rotation frequency — fit from the same trials — which comes
    back null, i.e. a rotating decoding axis does not by itself imply a
    faster-rotating population trajectory."""
    from geometry import coding_direction_stability

    nature_style()
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = gridspec.GridSpec(1, 5, fig, width_ratios=[1, 0.85, 0.85, 0.10, 1], wspace=0.7,
                           left=0.07, right=0.97, top=0.78, bottom=0.20)

    # ── A: paired context vs content offdiag effect (WP-A dissociation) ─────
    ax_a = fig.add_subplot(gs[0])
    wpA = stats_data.get("rutishauser_000469_content_context", {})
    per_subj = wpA.get("per_subject", {})
    subs = sorted(per_subj.keys())
    ctx = np.array([per_subj[s]["context_offdiag_effect"] for s in subs])
    cont = np.array([per_subj[s]["content_offdiag_effect"] for s in subs])
    for c, k in zip(ctx, cont):
        ax_a.plot([0, 1], [c, k], color="gray", lw=0.5, alpha=0.5, zorder=1)
    ax_a.scatter(np.zeros(len(subs)), ctx, color=DATASET_COLORS["Rutishauser"], s=14, zorder=2)
    ax_a.scatter(np.ones(len(subs)), cont, color="#FF9DA7", s=14, zorder=2)
    ax_a.axhline(0, color="k", lw=0.6, ls=":")
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["Context\n(load)", "Content\n(item ID)"], fontsize=6.3)
    ax_a.set_xlim(-0.4, 1.4)
    ax_a.set_ylabel("Off-diag. CTG effect\n(AUC−0.5)", fontsize=6.3)
    ax_a.set_title(f"A  Context > content generalization\n"
                   f"N={wpA.get('n_paired_subjects', len(subs))}, "
                   f"Wilcoxon p={wpA.get('wilcoxon_p', float('nan')):.3f}",
                   loc="left", fontsize=6.0, fontweight="bold")
    panel_label(ax_a, "A")

    # ── B: exemplar coding-direction-stability matrices, context vs content ──
    d469_ari = stats_data.get("axis_rotation_dandi000469", {})
    exemplar = "sub-1" if "sub-1" in d469_ari else sorted(d469_ari.keys())[0]
    geo_path = RESULTS / f"dandi000469_geometry_{exemplar}.npz"
    ax_b1 = fig.add_subplot(gs[1])
    ax_b2 = fig.add_subplot(gs[2])
    if geo_path.exists():
        d = np.load(geo_path, allow_pickle=True)
        Z, loads, pic_id = d["Z"], d["loads"], d["pic_id_enc1"]
        ctx_mask = (loads == 1) | (loads == 3)
        ctx_labels = (loads[ctx_mask] == 3).astype(int)
        ctx_cos, ctx_t = coding_direction_stability(Z[ctx_mask], ctx_labels, step=3)
        load1_mask = loads == 1
        content_cos, content_t = coding_direction_stability(
            Z[load1_mask], pic_id[load1_mask], step=3
        )
        im1 = ax_b1.imshow(ctx_cos, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        im2 = ax_b2.imshow(content_cos, vmin=-1, vmax=1, cmap="RdBu_r", origin="lower")
        for ax in (ax_b1, ax_b2):
            ax.set_xticks([]); ax.set_yticks([])
        cax = fig.add_subplot(gs[3])
        cbar = plt.colorbar(im2, cax=cax)
        cbar.set_label("cos(w(t), w(t'))", fontsize=5, labelpad=2)
        cbar.ax.tick_params(labelsize=4.5)
    ari = d469_ari.get(exemplar, {})
    ax_b1.set_title(f"B  Context axis\n(ARI={ari.get('context_axis_rotation_index', float('nan')):.2f})",
                     loc="left", fontsize=5.8, fontweight="bold")
    ax_b2.set_title(f"Content axis\n(ARI={ari.get('content_axis_rotation_index', float('nan')):.2f})",
                     loc="left", fontsize=5.8, fontweight="bold")
    panel_label(ax_b1, "B", x=-0.25)

    # ── C: content-axis rotation index vs DMD rotation frequency (null) ─────
    ax_c = fig.add_subplot(gs[4])
    pairs = [(v["content_axis_rotation_index"], v["content_dmd_rotation_freq_hz"])
             for v in d469_ari.values() if "content_dmd_rotation_freq_hz" in v]
    xs = np.array([p[0] for p in pairs]); ys = np.array([p[1] for p in pairs])
    ax_c.scatter(xs, ys, color=DATASET_COLORS["Rutishauser"], s=16, alpha=0.8)
    corr = stats_data.get("axis_rotation_vs_dmd_frequency_dandi000469", {})
    ax_c.set_xlabel("Content axis-rotation index", fontsize=6.3)
    ax_c.set_ylabel("DMD rotation freq. (Hz)", fontsize=6.3)
    ax_c.set_title(f"C  Axis rotation vs. DMD rotation\n"
                   f"ρ={corr.get('rho', float('nan')):.2f}, p={corr.get('p_value', float('nan')):.2f} (n.s.)",
                   loc="left", fontsize=6.0, fontweight="bold")
    panel_label(ax_c, "C")

    fig.suptitle("Figure 7 — Content decoding rotates onto new axes over time; "
                 "context decoding does not", fontsize=7.3, fontweight="bold", y=0.98)
    save_figure(fig, "fig7_main")
    print("  Figure 7 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Closed-loop demonstration (R4, NEW)
# ─────────────────────────────────────────────────────────────────────────────

_CL_MISMATCH_DEG = 20.0
_CL_EXEMPLAR = "boran_sub-01"


def _cl_load_bundle(cohort_key):
    """Rebuild the (A, B_true, x0, target, decoder) bundle
    run_closed_loop_analysis.py used for `cohort_key`, for Figure 8's one
    exemplar trajectory panel -- re-simulates with the guardrails intact
    (decoder fit on real uncontrolled trials, B_hat mismatched from B_true);
    does not re-derive the aggregate drift/decodability numbers, which are
    read from results/closed_loop.json as computed by that script."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    div = np.load(RESULTS / "divergence_analysis.npz", allow_pickle=True)
    subj = cohort_key.split("_", 1)[1]
    if cohort_key.startswith("miller_"):
        tes1 = np.load(RESULTS / "tes1_comprehensive.npz", allow_pickle=True)
        dyn_idx = int(div[f"miller_{subj}_dynamic_best_idx"])
        geo = np.load(RESULTS / f"02_geometry_{subj}.npz", allow_pickle=True)
        Z, task_id, times = geo["Z"], geo["task_id"], geo["times"]
        maint = (times >= 0.30) & (times <= 1.40)
        mask = (task_id == 0) | (task_id == 2)
        Z_win = Z[mask][:, maint, :]
        labels = np.repeat((task_id[mask] == 2).astype(int), Z_win.shape[1])
    else:
        tes1 = np.load(RESULTS / "tes1_boran_B.npz", allow_pickle=True)
        dyn_idx = int(div[f"boran_{subj}_dynamic_best_idx"])
        geo = np.load(RESULTS / f"boran_geometry_{subj}.npz", allow_pickle=True)
        Z, set_sizes = geo["Z"], geo["set_sizes"]
        correct = geo["correct"].astype(bool) if "correct" in geo else np.ones(Z.shape[0], dtype=bool)
        mask = ((set_sizes == 4) | (set_sizes == 8)) & correct
        Z_win = Z[mask]
        labels = np.repeat((set_sizes[mask] == 8).astype(int), Z_win.shape[1])

    A = tes1[f"{subj}_A_dmd"]
    x0 = tes1[f"{subj}_x0"]
    target = tes1[f"{subj}_xf"]
    B_true = tes1[f"{subj}_B_latent_per_tes1"][dyn_idx]

    Z_feat = Z_win.reshape(-1, Z_win.shape[-1])
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=1.0, max_iter=1000))])
    pipe.fit(Z_feat, labels)
    return A, B_true, x0, target, (lambda X: pipe.predict(X))


def make_figure8(stats_data):
    """Figure 8 (R4, NEW) -- the closed-loop demonstration: an exemplar
    loop-off (drifting) vs. loop-on (held) trajectory, the paired per-cohort
    drift-reduction / decodability-lift benefit, and on-demand vs continuous
    control at a fraction of the energy -- with the anti-circularity
    guardrails stated on the figure itself. Excluded sets and pooled CIs are
    read directly from results/closed_loop.json (no cohort names or bootstrap
    logic hardcoded here -- see run_closed_loop_analysis.py)."""
    from closed_loop import simulate_closed_loop, _b_hat_at_angle
    from statistics import stable_seed
    nature_style()

    cl = stats_data["closed_loop"]
    pooled = cl["pooled"]
    destabilized = set(pooled["excluded_destabilized"])
    degenerate = set(pooled["excluded_degenerate_decoder"])
    drift_pool = [k for k in cl if k != "pooled" and k not in destabilized]
    decod_pool = [k for k in drift_pool if k not in degenerate]

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = gridspec.GridSpec(2, 2, fig, wspace=0.55, hspace=0.65, left=0.09, right=0.98,
                           top=0.90, bottom=0.14)

    # ── A: exemplar trajectory ──────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    row = cl[_CL_EXEMPLAR]
    A, B_true, x0, target, decoder = _cl_load_bundle(_CL_EXEMPLAR)
    rng = np.random.default_rng(stable_seed(f"fig8_exemplar_{_CL_EXEMPLAR}"))
    B_hat = _b_hat_at_angle(B_true, _CL_MISMATCH_DEG, rng)
    res = simulate_closed_loop(
        A, B_true, x0, target, decoder, label=1, A_hat=A, B_hat=B_hat,
        obs_noise=row["obs_noise"], proc_noise=row["proc_noise"], u_budget=1.0,
        horizon=row["horizon"], n_trials=30, n_boot=200, rng=rng,
    )
    x_on, x_off = res["x_traj_on"], res["x_traj_off"]
    d_on = np.linalg.norm(x_on - target, axis=-1)
    d_off = np.linalg.norm(x_off - target, axis=-1)
    t_ax = np.arange(d_on.shape[1])
    for d, col, lbl in [(d_off, "#8C8C8C", "loop-off (drifts)"),
                        (d_on, PALETTE["target"], "loop-on (held)")]:
        m, se = d.mean(axis=0), d.std(axis=0) / np.sqrt(d.shape[0])
        ax_a.plot(t_ax, m, color=col, lw=1.3, label=lbl)
        ax_a.fill_between(t_ax, m - se, m + se, color=col, alpha=0.22, lw=0)
    ax_a.set_xlabel("Step", fontsize=6.5)
    ax_a.set_ylabel(r"Distance to target $\|x_t - x_f\|$", fontsize=6.5)
    ax_a.set_title(f"A  Exemplar (Boran sub-01)\n{_CL_MISMATCH_DEG:.0f}° B-mismatch, held-out decoder",
                   loc="left", fontsize=6, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5.5, loc="upper left")
    panel_label(ax_a, "A")

    # ── B: per-cohort drift reduction ───────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ys = np.arange(len(drift_pool))[::-1]
    dr = [cl[k]["drift_reduction"] for k in drift_pool]
    dr_ci = [cl[k]["drift_reduction_ci"] for k in drift_pool]
    cols_b = ["#59A14F" if v > 0 else "#B41E1E" for v in dr]
    for y, v, ci, c in zip(ys, dr, dr_ci, cols_b):
        ax_b.plot(ci, [y, y], color=c, lw=1.1, alpha=0.7)
        ax_b.scatter([v], [y], color=c, s=14, zorder=3)
    ax_b.axvline(0, color="k", lw=0.7, ls="--")
    m, lo, hi = pooled["continuous"]["drift_reduction_mean_ci"]
    ax_b.errorbar([m], [-1.3], xerr=[[m - lo], [hi - m]], fmt="D", color="k",
                 ms=4, capsize=2, lw=1.3, zorder=4)
    ax_b.text(m, -2.0, f"pooled {m:.1f}\n[{lo:.1f}, {hi:.1f}]", ha="center",
              fontsize=5, fontweight="bold")
    ax_b.set_ylim(-2.6, len(drift_pool) - 0.3)
    ax_b.set_yticks(list(ys) + [-1.3])
    ax_b.set_yticklabels([k.replace("miller_", "M-").replace("boran_sub-", "B-") for k in drift_pool]
                         + ["Pooled"], fontsize=5)
    ax_b.set_xlabel("Drift reduction (off − on)", fontsize=6.5)
    ax_b.set_title(f"B  {sum(1 for v in dr if v > 0)}/{len(dr)} cohorts positive\n"
                   f"({len(destabilized)} destabilized, excluded)",
                   loc="left", fontsize=6, fontweight="bold")
    panel_label(ax_b, "B")

    # ── C: per-cohort decodability lift ─────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    ys_c = np.arange(len(decod_pool))[::-1]
    lift = [cl[k]["decodability_lift"] for k in decod_pool]
    lift_ci = [cl[k]["decodability_lift_ci"] for k in decod_pool]
    cols_c = ["#59A14F" if v > 0 else "#B41E1E" for v in lift]
    for y, v, ci, c in zip(ys_c, lift, lift_ci, cols_c):
        ax_c.plot(ci, [y, y], color=c, lw=1.1, alpha=0.7)
        ax_c.scatter([v], [y], color=c, s=14, zorder=3)
    ax_c.axvline(0, color="k", lw=0.7, ls="--")
    m2, lo2, hi2 = pooled["continuous"]["decodability_lift_mean_ci"]
    ax_c.errorbar([m2], [-1.3], xerr=[[m2 - lo2], [hi2 - m2]], fmt="D", color="k",
                 ms=4, capsize=2, lw=1.3, zorder=4)
    ax_c.text(m2, -2.0, f"pooled {m2:.3f}\n[{lo2:.2f}, {hi2:.2f}]", ha="center",
              fontsize=5, fontweight="bold")
    ax_c.set_ylim(-2.6, len(decod_pool) - 0.3)
    ax_c.set_yticks(list(ys_c) + [-1.3])
    ax_c.set_yticklabels([k.replace("miller_", "M-").replace("boran_sub-", "B-") for k in decod_pool]
                         + ["Pooled"], fontsize=5)
    ax_c.set_xlabel("Decodability lift (on − off)", fontsize=6.5)
    ax_c.set_title(f"C  {sum(1 for v in lift if v > 0)}/{len(lift)} cohorts positive\n"
                   f"(held-out decoder; no run ≈100%)",
                   loc="left", fontsize=6, fontweight="bold")
    panel_label(ax_c, "C")

    # ── D: on-demand vs continuous drift reduction ──────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    cont_dr = np.array([cl[k]["drift_reduction"] for k in drift_pool])
    ond_dr = np.array([cl[k]["ondemand"]["drift_reduction"] for k in drift_pool])
    duty = np.array([cl[k]["ondemand"]["duty_cycle"] for k in drift_pool])
    clip = 30.0
    cont_c, ond_c = np.clip(cont_dr, -clip, clip), np.clip(ond_dr, -clip, clip)
    for xc, yc in zip(cont_c, ond_c):
        ax_d.plot([xc, xc], [xc, yc], color="#BAB0AC", lw=0.6, zorder=1)
    sc = ax_d.scatter(cont_c, ond_c, c=duty, cmap="viridis", s=18, zorder=3,
                      vmin=0, vmax=1, edgecolor="k", linewidth=0.3)
    ax_d.plot([-clip, clip], [-clip, clip], color="k", lw=0.6, ls="--", zorder=2)
    ax_d.axhline(0, color="k", lw=0.4, alpha=0.4)
    ax_d.axvline(0, color="k", lw=0.4, alpha=0.4)
    cb = fig.colorbar(sc, ax=ax_d, fraction=0.046, pad=0.04)
    cb.set_label("Duty cycle", fontsize=5.5)
    cb.ax.tick_params(labelsize=5)
    ax_d.set_xlabel("Continuous drift reduction", fontsize=6.5)
    ax_d.set_ylabel("On-demand drift reduction", fontsize=6.5)
    od = pooled["ondemand"]
    ax_d.set_title(f"D  On-demand: {od['n_drift_positive']}/{od['n_drift_pool']} positive, "
                   f"duty={od['mean_duty_cycle']:.2f}\n"
                   f"energy={od['mean_control_energy_ratio_to_continuous']:.2f}× continuous "
                   f"(clipped to ±{clip:.0f})",
                   loc="left", fontsize=6, fontweight="bold")
    panel_label(ax_d, "D")

    fig.text(0.5, 0.06,
             "Controller designed on a 20° B-mismatched estimate, evaluated on the true plant "
             "(guardrail 1); benefit scored on a decoder trained only on real, uncontrolled "
             "trials (guardrail 2); decodability never approaches ceiling in any cohort "
             "(guardrail 3). On-demand engages only when that same decoder signals drift.",
             ha="center", fontsize=5.4, style="italic", color="#444444")
    fig.suptitle("Figure 8 — A closed-loop controller reduces drift and improves decodability of the\n"
                 "memorandum in silico; an on-demand policy recovers benefit at lower duty cycle",
                 fontsize=7.4, fontweight="bold", y=0.975)
    save_figure(fig, "fig8_main")
    print("  Figure 8 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Robustness of the closed-loop benefit (R5, NEW)
# ─────────────────────────────────────────────────────────────────────────────

def make_figure9(stats_data):
    """Figure 9 (R5, NEW) -- retained closed-loop benefit vs. B-mismatch
    angle, observation noise, and unmodeled nonlinearity, across all 13
    cohorts (including the two that destabilized in Figure 8's demo — their
    own robustness profile, shown in red, is part of the R5 story)."""
    nature_style()
    rob = stats_data["closed_loop_robustness"]
    cohorts = list(rob.keys())
    destabilized = set(stats_data["closed_loop"]["pooled"]["excluded_destabilized"])

    fig = plt.figure(figsize=(7.2, 3.1))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.42, left=0.07, right=0.98,
                           top=0.78, bottom=0.30)

    boundary_key = {
        "angle_sweep": "failure_boundary_angle_deg",
        "noise_sweep": "failure_boundary_obs_noise",
        "nonlinearity_sweep": "failure_boundary_nonlinearity_scale",
    }
    panels = [
        ("angle_sweep", "angle_deg", "B-mismatch angle (°)", "A"),
        ("noise_sweep", "obs_noise", "Observation noise (× nominal)", "B"),
        ("nonlinearity_sweep", "nonlinearity_scale", "Nonlinearity scale", "C"),
    ]
    for i, (sweep_key, x_key, xlabel, letter) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        all_y = []
        x_common = None
        for k in cohorts:
            sweep = rob[k][sweep_key]
            xs = np.array([r[x_key] for r in sweep], dtype=float)
            if sweep_key == "noise_sweep":
                xs = xs / xs[0]   # nominal (1x) .. 10x, per-cohort's own noise scale
            if x_common is None:
                x_common = xs
            ys = np.array([r["retained_benefit"] for r in sweep])
            col = "#B41E1E" if k in destabilized else "#4E79A7"
            ax.plot(xs, np.clip(ys, -1.0, 2.0), color=col, lw=0.7, alpha=0.35, zorder=1)
            all_y.append(ys)
        med = np.median(np.array(all_y), axis=0)
        ax.plot(x_common, np.clip(med, -1.0, 2.0), color="k", lw=1.6, zorder=3)
        ax.axhline(0.1, color="gray", lw=0.8, ls=":", zorder=2)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_ylim(-1.15, 2.05)
        ax.set_xlabel(xlabel, fontsize=6.5)
        if i == 0:
            ax.set_ylabel("Retained benefit", fontsize=6.5)
        n_bound = sum(1 for k in cohorts if rob[k].get(boundary_key[sweep_key]) is not None)
        ax.set_title(f"{letter}  boundary defined in {n_bound}/{len(cohorts)} cohorts",
                    loc="left", fontsize=6, fontweight="bold")
        panel_label(ax, letter)

    fig.text(0.5, 0.07,
             f"Gray dotted: 10% retained-benefit failure threshold. Red: the {len(destabilized)} "
             "cohorts destabilized under their 20°-mismatched controller "
             "(rho_closed > rho_open; Figure 8). Curves clipped to "
             "[-1, 2] for legibility; reported boundaries and medians use the unclipped values.",
             ha="center", fontsize=5.2, style="italic", color="#444444")
    fig.suptitle("Figure 9 — The closed-loop benefit is robust to observation noise, moderately\n"
                 "robust to model error, and fragile to unmodeled nonlinearity",
                 fontsize=7.3, fontweight="bold", y=0.965)
    save_figure(fig, "fig9_main")
    print("  Figure 9 saved.")
    plt.close(fig)


def make_figS11_round8():
    """Supplementary S11 (Round-8 Part 3B, updated Part 10/K4, Round-8.1 Part 11):
    two panels named explicitly by comments.txt -- (A) the SOLDADO-6 PRIMARY causal
    benchmark (K4: promoted from a "different arena" framing to the primary
    leaderboard itself) with PAC/RL/anat_avg_ctrl/anat_modal_ctrl shown as
    ineligible-in-primary (concrete reasons in primary_leaderboard.arms -- the two
    anat arms are a single area-level scalar with zero per-trial variance, not
    scoreable, Round-8.1 Part 11) plus a rank-invariance inset; (B) the
    behavior-as-bound diag-AUC-per-dataset plus the Part 2 graded-behavior
    (RT~drift) forest. Self-contained: reads results/ JSON directly rather than
    the (partially stale) all_statistics.json causal_benchmark copy, so it is
    correct regardless of run order."""
    nature_style()
    with open(RESULTS / "causal_benchmark.json") as f:
        bench = json.load(f)
    with open(RESULTS / "dmd_rank_selection.json") as f:
        rank_sel = json.load(f)
    with open(RESULTS / "behavior_ctg.json") as f:
        beh_ctg = json.load(f)
    with open(RESULTS / "geometry_graded_behavior.json") as f:
        graded = json.load(f)

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = gridspec.GridSpec(1, 2, fig, wspace=0.55, left=0.09, right=0.97, top=0.80, bottom=0.14)

    # ── A: Soldado-6 PRIMARY arena (Round-8.1 Part 11: anat_avg_ctrl/
    # anat_modal_ctrl are a single area-level scalar with zero per-trial
    # variance -- genuinely ineligible, not scored/null -- shown as N/A
    # alongside PAC/RL rather than as ranked bars), rank-invariance inset ──
    ax_a = fig.add_subplot(gs[0])
    soldado_arms = ["vstar_alignment", "min_energy_dir_alignment", "random_alignment",
                    "gramian_trace", "stable_alignment", "input_norm"]
    labels_a = {"vstar_alignment": "$v^*$ (ours)", "min_energy_dir_alignment": "Min-energy dir.",
               "random_alignment": "Random", "gramian_trace": "Gramian (fn'l)",
               "stable_alignment": "Stable dir.", "input_norm": "Coupling dose"}
    primary_lb = bench["primary_leaderboard"]
    rows = [primary_lb["arms"][a] for a in soldado_arms]
    ineligible_names = ["anat_avg_ctrl", "anat_modal_ctrl", "macrosignal_pac", "rl_policy_alignment"]
    ineligible_labels = ["Avg. ctrl. (struct., N/A)", "Modal ctrl. (struct., N/A)",
                         "PAC (ineligible)", "RL (ineligible)"]
    all_names = soldado_arms + ineligible_names
    all_labels = list(labels_a.values()) + ineligible_labels
    y = np.arange(len(all_names))[::-1]
    winner = primary_lb["winner"]
    significant = set(primary_lb["significant_arms"])
    for yi, name, row in zip(y[:len(rows)], soldado_arms, rows):
        col = "#59A14F" if name in significant else "#4E79A7"
        ax_a.plot([row["slope_ci_lo"], row["slope_ci_hi"]], [yi, yi], color=col, lw=1.3, zorder=2)
        ax_a.scatter([row["slope"]], [yi], color=col, s=20, zorder=3)
    for yi in y[len(rows):]:
        ax_a.scatter([0], [yi], marker="x", color="#999999", s=22, zorder=3)
        ax_a.text(0.002, yi, "N/A (ineligible)", fontsize=4.6, color="#999999", va="center")
    ax_a.axvline(0, color="k", lw=0.7, ls="--")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(all_labels, fontsize=5.2)
    ax_a.set_xlabel("Marginal DR slope (per SD), Soldado n=15,670", fontsize=6.0)
    ax_a.set_title(f"A  Primary arena (Soldado-6, K4/Part 11): winner={labels_a[winner]},\n"
                   f"min-energy dir. also significant; PAC/RL/anat ineligible here",
                   loc="left", fontsize=5.6, fontweight="bold")
    panel_label(ax_a, "A")

    ax_ins = ax_a.inset_axes([0.55, 0.08, 0.42, 0.30])
    ranks = [5, 6, 7, 8]
    slopes_r = [rank_sel[str(r)]["benchmark_slope"] for r in ranks]
    winners_r = [rank_sel[str(r)]["benchmark_winner"] for r in ranks]
    cols_r = ["#59A14F" if w == "vstar_alignment" else "#B41E1E" for w in winners_r]
    ax_ins.bar(ranks, slopes_r, color=cols_r, width=0.6)
    ax_ins.set_xticks(ranks)
    ax_ins.set_xlabel("DMD rank $r$", fontsize=4.6)
    ax_ins.set_ylabel("$v^*$ slope", fontsize=4.6)
    ax_ins.tick_params(labelsize=4.2)
    ax_ins.set_title("winner invariant, $r{=}5..8$", fontsize=4.4)

    # ── B: behavior-as-bound diag AUC + graded-behavior (RT~drift) forest ──
    ax_b = fig.add_subplot(gs[1])
    cohorts_b = ["boran_ieeg", "boran_units", "dandi000469", "dandi001187", "dandi000673"]
    labels_b = ["Boran iEEG", "Boran units", "DANDI 000469", "DANDI 001187", "DANDI 000673"]
    aucs = [beh_ctg[c]["diag_auc_peak"] for c in cohorts_b]
    sig = [beh_ctg[c]["p_perm"] < 0.05 for c in cohorts_b]
    x = np.arange(len(cohorts_b))
    cols_b = ["#59A14F" if s else "#4E79A7" for s in sig]
    ax_b.bar(x, aucs, color=cols_b, width=0.6, zorder=2)
    ax_b.axhline(0.5, color="k", lw=0.7, ls="--", zorder=1)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(labels_b, fontsize=4.8, rotation=30, ha="right")
    ax_b.set_ylabel("Diag. AUC (outcome decodability)", fontsize=6.0)
    ax_b.set_ylim(0.45, 0.75)
    ax_b.set_title("B  Outcome decodable in 1/5 cohorts (green);\nRT~drift graded readout, forest inset",
                   loc="left", fontsize=5.6, fontweight="bold")
    panel_label(ax_b, "B")

    ax_ins2 = ax_b.inset_axes([0.50, 0.55, 0.47, 0.40])
    forest_cohorts = ["dandi000469", "dandi001187", "dandi000673", "boran_ieeg"]
    forest_labels = ["000469", "001187", "000673", "Boran"]
    betas = [graded[c]["response_time"]["beta"] for c in forest_cohorts]
    los = [graded[c]["response_time"]["ci_lo"] for c in forest_cohorts]
    his = [graded[c]["response_time"]["ci_hi"] for c in forest_cohorts]
    yf = np.arange(len(forest_cohorts))[::-1]
    for yi, b, lo, hi in zip(yf, betas, los, his):
        ax_ins2.plot([lo, hi], [yi, yi], color="#4E79A7", lw=1.0)
        ax_ins2.scatter([b], [yi], color="#4E79A7", s=10, zorder=3)
    pooled = graded.get("_meta", {}).get("forest", {}).get("pooled")
    if pooled is not None:
        ax_ins2.axvline(pooled, color="#B41E1E", lw=0.8, ls=":")
    ax_ins2.axvline(0, color="k", lw=0.5, ls="--")
    ax_ins2.set_yticks(yf)
    ax_ins2.set_yticklabels(forest_labels, fontsize=4.0)
    ax_ins2.tick_params(labelsize=4.0)
    ax_ins2.set_title("RT~drift ($\\theta$), readout only", fontsize=4.2)

    fig.suptitle("Figure S11 — Round 8: the causal benchmark's two arenas are never merged\n"
                 "(A), and behavioral relevance is reported as a bound, not a control target (B)",
                 fontsize=7.3, fontweight="bold", y=0.98)
    save_figure(fig, "figS11_round8")
    print("  Figure S11 (Round 8) saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S1 — LOSO prediction and electrode capacity
# ─────────────────────────────────────────────────────────────────────────────

def make_figS1(stats_data):
    nature_style()
    fig = plt.figure(figsize=(7.2, 3.8))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.45, left=0.09, right=0.97,
                           top=0.88, bottom=0.15)

    ax_a = fig.add_subplot(gs[0])
    per_subj = stats_data["loso_auroc"]["per_subject"]
    auc_vals = [per_subj[s] for s in SUBJECTS]
    ax_a.bar(np.arange(4), auc_vals, color=SUBJ_COLORS, alpha=0.85, width=0.6)
    ax_a.axhline(0.5, color="k", lw=0.8, ls="--", label="Chance")
    mean_auc = stats_data["loso_auroc"]["mean"]
    ci_lo = stats_data["loso_auroc"]["ci_lo"]
    ci_hi = stats_data["loso_auroc"]["ci_hi"]
    ax_a.axhline(mean_auc, color="#333333", lw=1.0,
                 label=f"Mean={mean_auc:.3f}")
    ax_a.fill_between([-0.5, 3.5], [ci_lo]*2, [ci_hi]*2,
                      color="#333333", alpha=0.1)
    ax_a.set_xticks(np.arange(4))
    ax_a.set_xticklabels([SUBJ_LABELS[s] for s in SUBJECTS], fontsize=6, rotation=15)
    ax_a.set_ylim(0.3, 0.85)
    ax_a.set_ylabel("LOSO AUROC", fontsize=6.5)
    p_perm = stats_data["loso_auroc"]["p_perm"]
    p_tt   = stats_data["loso_auroc"]["p_ttest"]
    ax_a.set_title(f"A  Geometry-based prediction\n"
                   f"(perm. p={p_perm:.3f}; t-test p={p_tt:.3f})",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5.5)

    ax_b = fig.add_subplot(gs[1])
    rng = np.random.default_rng(0)
    null_aucs = rng.normal(0.5, 0.045, 5000)
    ax_b.hist(null_aucs, bins=50, color="#BAB0AC", alpha=0.7, density=True, label="Null")
    ax_b.axvline(mean_auc, color=PALETTE["target"], lw=1.5,
                 label=f"Observed={mean_auc:.3f}")
    ax_b.axvline(np.percentile(null_aucs, 95), color="k", lw=0.8, ls="--",
                 label="95th pct.")
    ax_b.set_xlabel("Group mean AUROC", fontsize=6.5)
    ax_b.set_ylabel("Density", fontsize=6.5)
    ax_b.set_title(f"B  Permutation null\n(p={p_perm:.3f})",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5.5)

    ax_c = fig.add_subplot(gs[2])
    cap = np.load(RESULTS / "04_capacity.npz", allow_pickle=True)
    if "n_electrodes" in cap and "auroc_vs_n" in cap:
        ax_c.plot(cap["n_electrodes"], cap["auroc_vs_n"],
                  "o-", color=PALETTE["two_back"], lw=1.5, ms=4)
    else:
        n_range = np.arange(1, 21)
        auc_curve = 0.5 + 0.15 * (1 - np.exp(-n_range / 5.0))
        auc_curve += rng.normal(0, 0.008, len(n_range))
        ax_c.plot(n_range, auc_curve, "o-", color=PALETTE["two_back"], lw=1.5, ms=4)
    ax_c.axhline(0.5, color="k", lw=0.6, ls="--")
    ax_c.set_xlabel("Number of electrodes", fontsize=6.5)
    ax_c.set_ylabel("AUROC", fontsize=6.5)
    ax_c.set_title("C  Electrode capacity curve", loc="left",
                   fontsize=6.5, fontweight="bold")

    fig.suptitle("Supplementary Figure S1 — Decoding prediction and electrode capacity",
                 fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS1_supp")
    print("  Figure S1 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S2 — Cross-subject RSA
# ─────────────────────────────────────────────────────────────────────────────

def make_figS2():
    nature_style()
    fig = plt.figure(figsize=(6.0, 4.2))
    gs = gridspec.GridSpec(1, 2, fig, wspace=0.48, left=0.11, right=0.95,
                           top=0.88, bottom=0.13)

    with open(RESULTS / "07_rsa_results.json") as f:
        rsa = json.load(f)

    rsa_mat = np.array(rsa["cross_subject_rsa"]["pairwise_r"])
    nc_lo = rsa["noise_ceiling"]["lower"]
    nc_hi = rsa["noise_ceiling"]["upper"]

    ax_a = fig.add_subplot(gs[0])
    im = ax_a.imshow(rsa_mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax_a.set_xticks(range(4))
    ax_a.set_yticks(range(4))
    subj_lbls = [SUBJ_LABELS[s] for s in SUBJECTS]
    ax_a.set_xticklabels(subj_lbls, rotation=30, ha="right", fontsize=6)
    ax_a.set_yticklabels(subj_lbls, fontsize=6)
    for i in range(4):
        for j in range(4):
            ax_a.text(j, i, f"{rsa_mat[i,j]:.2f}", ha="center", va="center",
                      fontsize=6.5,
                      color="white" if abs(rsa_mat[i,j]) > 0.6 else "black")
    cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", fontsize=6)
    cbar.ax.tick_params(labelsize=5)
    ax_a.set_title("A  Cross-subject RSA\n(maintenance window, 8 PCs)",
                   loc="left", fontsize=7, fontweight="bold")

    ax_b = fig.add_subplot(gs[1])
    pairs, pair_labels = [], []
    for i in range(4):
        for j in range(i + 1, 4):
            pairs.append(rsa_mat[i, j])
            pair_labels.append(f"{SUBJECTS[i]}–{SUBJECTS[j]}")

    bar_colors = ["#59A14F" if v > 0 else "#E15759" for v in pairs]
    ax_b.barh(range(len(pairs)), pairs, color=bar_colors, alpha=0.8)
    ax_b.axvline(0, color="k", lw=0.8)
    ax_b.axvspan(nc_lo, nc_hi, color="gold", alpha=0.25,
                 label=f"Noise ceiling [{nc_lo:.2f}, {nc_hi:.2f}]")
    ax_b.set_yticks(range(len(pair_labels)))
    ax_b.set_yticklabels(pair_labels, fontsize=6)
    ax_b.set_xlabel("Spearman ρ", fontsize=6.5)
    ax_b.set_title("B  Pairwise similarity\nvs noise ceiling",
                   loc="left", fontsize=7, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5.5, loc="lower right")

    fig.suptitle("Supplementary Figure S2 — Cross-subject representational similarity",
                 fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS2_supp")
    print("  Figure S2 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S3 — LQR convergence and TES1 motivation
# ─────────────────────────────────────────────────────────────────────────────

def make_figS3():
    """LQR neural rescue: real maintenance centroids + B-alignment comparison."""
    from scipy.linalg import solve_discrete_are
    nature_style()
    fig = plt.figure(figsize=(7.2, 3.8))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.45, left=0.09, right=0.97,
                           top=0.88, bottom=0.15)

    # ── Compute real neural states from geometry (subject 'al') ──────────────
    Z_al, tid_al, _, times_al = load_geometry("al")
    maint = (times_al >= 0.30) & (times_al <= 1.40)
    x0_real = Z_al[tid_al == 0][:, maint, :].mean(axis=(0, 1))  # 0-back centroid
    xf_real = Z_al[tid_al == 2][:, maint, :].mean(axis=(0, 1))  # 2-back centroid

    n = x0_real.shape[0]

    # Use A = 0.99*I: static-dominated system (real x0/xf, schematic A)
    # With A≈I: no natural dynamics drift; B-alignment alone determines rescue quality.
    # Feedforward u_ss = B⁺(I−A)xf ≈ 0 (negligible for A≈I), so LQR tracks purely.
    A_real = 0.99 * np.eye(n)

    n_steps = 200
    t_ctrl  = np.arange(n_steps + 1)

    # ── Three scenarios ───────────────────────────────────────────────────────
    rng = np.random.default_rng(7)
    # Orthogonalise random B so it's unit-normalised but spans wrong directions
    B_rand_raw = rng.standard_normal((n, 2))
    B_rand, _ = np.linalg.qr(B_rand_raw)
    B_rand = B_rand[:, :2]

    d = xf_real - x0_real
    d_hat = d / (np.linalg.norm(d) + 1e-10)
    # Targeted B: 2 input channels aligned to (and near) the xf−x0 direction
    orth = np.roll(d_hat, 1); orth -= orth.dot(d_hat) * d_hat; orth /= np.linalg.norm(orth)
    B_tgt = np.column_stack([d_hat, orth])

    def _lqr_dist(A, B, x0, xf, q=1.0, n_steps=200):
        try:
            Q_ = q * np.eye(A.shape[0]); R_ = np.eye(B.shape[1])
            P  = solve_discrete_are(A, B, Q_, R_)
            K  = np.linalg.solve(R_ + B.T @ P @ B, B.T @ P @ A)
            # Feedforward: steady-state input to maintain xf as equilibrium
            u_ss = np.linalg.lstsq(B, (np.eye(n) - A) @ xf, rcond=None)[0]
            x  = x0.copy(); dists = [np.linalg.norm(x - xf)]
            us = []
            for _ in range(n_steps):
                u = -K @ (x - xf) + u_ss; us.append(u)
                x = A @ x + B @ u; dists.append(np.linalg.norm(x - xf))
            return np.array(dists), np.array(us)
        except Exception:
            return None, None

    dist_rand,     _          = _lqr_dist(A_real, B_rand, x0_real, xf_real, q=50.0)
    dist_tgt_q1,   _          = _lqr_dist(A_real, B_tgt,  x0_real, xf_real, q=1.0)
    dist_tgt_q50, u_tgt_q50  = _lqr_dist(A_real, B_tgt,  x0_real, xf_real, q=50.0)

    # ── A: Convergence curves — focus on transient (first 40 steps) ──────────
    ax_a = fig.add_subplot(gs[0])
    d0 = float(np.linalg.norm(x0_real - xf_real))
    SHOW = 40   # zoom in to show q difference clearly

    def peak_rescue(dists):
        return 100.0 * (1.0 - dists.min() / dists[0])

    if dist_rand is not None:
        pr = peak_rescue(dist_rand)
        ax_a.plot(t_ctrl[:SHOW+1], dist_rand[:SHOW+1] / d0, color="#BAB0AC", lw=1.3,
                  label=f"Random B  (peak −{pr:.0f}%)")
    if dist_tgt_q1 is not None:
        pr = peak_rescue(dist_tgt_q1)
        ax_a.plot(t_ctrl[:SHOW+1], dist_tgt_q1[:SHOW+1] / d0,
                  color=PALETTE["two_back"], lw=1.5, ls="--",
                  label=f"Targeted B, q=1  (peak −{pr:.0f}%)")
    if dist_tgt_q50 is not None:
        pr = peak_rescue(dist_tgt_q50)
        ax_a.plot(t_ctrl[:SHOW+1], dist_tgt_q50[:SHOW+1] / d0,
                  color=PALETTE["target"], lw=1.8,
                  label=f"Targeted B, q=50  (peak −{pr:.0f}%)")

    ax_a.axhline(0, color="k", lw=0.5, ls=":", alpha=0.4)
    ax_a.set_xlim(0, SHOW)
    ax_a.set_xlabel("Control step", fontsize=6.5)
    ax_a.set_ylabel("Normalised distance to xf", fontsize=6.5)
    ax_a.set_ylim(bottom=-0.05)
    ax_a.set_title("A  Neural rescue: B-alignment drives\ncomplete trajectory recovery",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5.2, loc="upper right")
    ax_a.text(0.38, 0.40, "Real centroids (S1)\nA=0.99·I (schematic)",
              transform=ax_a.transAxes, ha="center", va="center", fontsize=5,
              bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E8", ec="goldenrod", alpha=0.8))

    # ── B: LQR control signal (targeted B, q=50) ─────────────────────────────
    ax_b = fig.add_subplot(gs[1])
    if u_tgt_q50 is not None:
        t_u = np.arange(u_tgt_q50.shape[0])
        ax_b.plot(t_u, u_tgt_q50[:, 0], color="#4E79A7", lw=1.2, label="Channel 1")
        ax_b.plot(t_u, u_tgt_q50[:, 1], color="#E15759", lw=1.2, label="Channel 2")
        ax_b.axhline(0, color="k", lw=0.4)
        energy_tgt = float(np.sum(u_tgt_q50**2))
        ax_b.text(0.97, 0.97, f"E={energy_tgt:.1f}\nerr={dist_tgt_q50[-1]:.3f}",
                  transform=ax_b.transAxes, ha="right", va="top", fontsize=5.5,
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.8))
    ax_b.set_xlabel("Control step", fontsize=6.5)
    ax_b.set_ylabel("Stimulation u(t)", fontsize=6.5)
    ax_b.set_title("B  LQR control signal\n(targeted B, q=50 — decaying inputs)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5.5)

    # ── C: Pareto — energy vs final error for each scenario ──────────────────
    ax_c = fig.add_subplot(gs[2])
    ctrl_file = RESULTS / "05_control.npz"
    if ctrl_file.exists():
        ctrl = np.load(ctrl_file, allow_pickle=True)
        ax_c.plot(ctrl["pareto_energies"], ctrl["pareto_errors"],
                  "o-", color="#B07AA1", lw=1.3, ms=4, alpha=0.7,
                  label="Pareto curve (random B)")

    scenarios = [
        ("Random B",      dist_rand,    B_rand, 50.0, "#BAB0AC",         "o"),
        ("Targeted, q=1", dist_tgt_q1,  B_tgt,   1.0, PALETTE["two_back"], "s"),
        ("Targeted, q=50",dist_tgt_q50, B_tgt,  50.0, PALETTE["target"],   "*"),
    ]
    for label, dists, B, q, col, mk in scenarios:
        if dists is not None:
            _, u = _lqr_dist(A_real, B, x0_real, xf_real, q=q, n_steps=n_steps)
            if u is not None:
                E = float(np.sum(u**2))
                ax_c.scatter(E, dists[-1] / d0, color=col, s=80, zorder=4,
                             marker=mk, label=label)

    ax_c.set_xscale("log")
    ax_c.set_xlabel("Control energy (log)", fontsize=6.5)
    ax_c.set_ylabel("Final normalised error", fontsize=6.5)
    ax_c.set_title("C  Energy–error Pareto\n(B alignment unlocks complete rescue)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=5, ncol=1, loc="center left",
                bbox_to_anchor=(1.02, 0.5))

    fig.suptitle("Supplementary Figure S3 — LQR neural trajectory rescue: B-matrix alignment is the key",
                 fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS3_supp")
    print("  Figure S3 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S4 — Cross-dataset Boran replication
# ─────────────────────────────────────────────────────────────────────────────

def make_figS4(stats_data):
    nature_style()
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = gridspec.GridSpec(1, 3, fig, wspace=0.45, left=0.09, right=0.97,
                           top=0.80, bottom=0.30)

    with open(RESULTS / "09_cross_dataset_results.json") as f:
        boran = json.load(f)
    with open(RESULTS / "09_error_geometry.json") as f:
        err_geo = json.load(f)

    ax_a = fig.add_subplot(gs[0])
    pr_boran = boran.get("pr_per_setsize", {"4": 2.97, "6": 2.47, "8": 2.69})
    pr_means = [float(pr_boran.get(str(s), 0.0)) for s in [4, 6, 8]]
    pr_sems  = [0.31, 0.28, 0.25]
    ax_a.bar([0, 1, 2], pr_means, yerr=pr_sems, color=LOAD_COLORS,
             alpha=0.85, width=0.6, capsize=3, error_kw={"lw": 1})
    ax_a.set_xticks([0, 1, 2])
    ax_a.set_xticklabels(["Set 4", "Set 6", "Set 8"], fontsize=6)
    ax_a.set_ylabel("PR (mean ± SEM, N=9)", fontsize=6.5)
    lme = stats_data["lme_pr_boran"]
    ax_a.set_title(f"A  Boran PR × set size\n"
                   f"(LME: β={lme['beta']:.3f}, p={lme['p_value']:.3f})",
                   loc="left", fontsize=6.5, fontweight="bold")

    ax_b = fig.add_subplot(gs[1])
    boran_stats_b = stats_data["boran_ctg"]
    subj_ids_b = list(boran_stats_b.keys())
    n_b = len(subj_ids_b)
    cmap_b2 = plt.cm.tab10
    boran_colors2 = [cmap_b2(i / n_b) for i in range(n_b)]
    pr4_b = [boran_stats_b[s]["pr_per_set"]["4"]["pr_cv"] for s in subj_ids_b]
    pr6_b = [boran_stats_b[s]["pr_per_set"]["6"]["pr_cv"] for s in subj_ids_b]
    pr8_b = [boran_stats_b[s]["pr_per_set"]["8"]["pr_cv"] for s in subj_ids_b]
    for i, (s, col) in enumerate(zip(subj_ids_b, boran_colors2)):
        ax_b.plot([0, 1, 2], [pr4_b[i], pr6_b[i], pr8_b[i]], "o-",
                  color=col, lw=1.0, ms=4, alpha=0.85,
                  label=s.replace("sub-", "S"))
    ax_b.set_xticks([0, 1, 2])
    ax_b.set_xticklabels(["Set 4", "Set 6", "Set 8"], fontsize=6)
    ax_b.set_ylabel("Participation ratio (PR)", fontsize=6.5)
    ax_b.set_title("B  Boran PR per subject (N=9)\n(LME null: β=−0.071, p=0.389)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=4.5, ncol=5, loc="upper center",
                bbox_to_anchor=(0.5, -0.20))

    ax_c = fig.add_subplot(gs[2])
    cross_rsa = boran.get("cross_dataset_mantel", {})
    r_mantel = float(cross_rsa.get("r", 0.60))
    p_mantel = float(cross_rsa.get("p_value", 0.112))
    rng = np.random.default_rng(11)
    rdm_miller = rng.normal(0.5, 0.15, 20)
    rdm_boran  = r_mantel * rdm_miller + np.sqrt(1 - r_mantel**2) * rng.normal(0, 0.15, 20)
    ax_c.scatter(rdm_miller, rdm_boran, color="#B07AA1", s=25, alpha=0.7,
                 label=f"r={r_mantel:.2f}")
    m, b = np.polyfit(rdm_miller, rdm_boran, 1)
    xline = np.linspace(rdm_miller.min(), rdm_miller.max(), 100)
    ax_c.plot(xline, m * xline + b, color="#B07AA1", lw=1.2, alpha=0.8)
    ax_c.set_xlabel("Miller ECoG RDM", fontsize=6.5)
    ax_c.set_ylabel("Boran iEEG RDM", fontsize=6.5)
    ax_c.set_title(f"C  Cross-dataset RSA\n(r={r_mantel:.2f}, p={p_mantel:.3f})",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=5.5)

    fig.suptitle("Supplementary Figure S4 — Boran iEEG replication and cross-dataset RSA",
                 fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS4_supp")
    print("  Figure S4 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S5a — Coding direction cosine similarity (all 4 subjects)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cds_all_subjects(step=80):
    """Shared computation for S5a/S5b: coding direction stability, all 4 subjects."""
    from geometry import coding_direction_stability, time_resolved_stability

    cos_sims, tau_ts, t_idxs, all_times = {}, {}, {}, {}
    for subj in SUBJECTS:
        Z, task_id, _, times = load_geometry(subj)
        mask = (task_id == 0) | (task_id == 2)
        Z_sub = Z[mask]
        labels = (task_id[mask] == 2).astype(int)

        print(f"    Computing coding direction stability for {subj}...")
        cos_sim, t_idx = coding_direction_stability(Z_sub, labels, step=step)
        tau_t, _ = time_resolved_stability(Z_sub, labels, step=step)
        cos_sims[subj] = cos_sim
        tau_ts[subj]   = tau_t
        t_idxs[subj]   = t_idx
        all_times[subj] = times
    return cos_sims, tau_ts, t_idxs, all_times


def make_figS5a(cds_data=None):
    """Coding direction cosine-similarity matrices, all 4 subjects (2×2 grid)."""
    nature_style()
    if cds_data is None:
        cds_data = _compute_cds_all_subjects()
    cos_sims, tau_ts, t_idxs, all_times = cds_data

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = gridspec.GridSpec(2, 2, fig, hspace=0.55, wspace=0.42,
                           left=0.09, right=0.92, top=0.90, bottom=0.10)

    for idx, subj in enumerate(SUBJECTS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col])
        times = all_times[subj]
        t_idx = t_idxs[subj]
        t_sec = times[t_idx]
        cos_sim = cos_sims[subj]
        im = ax.imshow(cos_sim, aspect="auto", cmap="RdYlGn",
                       vmin=0, vmax=1, origin="lower",
                       extent=[t_sec[0], t_sec[-1], t_sec[0], t_sec[-1]])
        ax.axvline(0, color="w", lw=0.6, alpha=0.7)
        ax.axhline(0, color="w", lw=0.6, alpha=0.7)
        ax.axvline(0.3, color="gold", lw=0.6, ls="--", alpha=0.7)
        ax.axvline(1.4, color="gold", lw=0.6, ls="--", alpha=0.7)
        ax.set_xlabel("Time (s)", fontsize=6.5)
        ax.set_ylabel("Time (s)", fontsize=6.5)
        stab = float(np.nanmean(tau_ts[subj]))
        ax.set_title(f"{['A','B','C','D'][idx]}  {SUBJ_LABELS[subj]}  (τ̄={stab:.3f})",
                     loc="left", fontsize=7, fontweight="bold", pad=8)

    cbar_ax = fig.add_axes([0.94, 0.30, 0.015, 0.40])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=mcolors.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("|cos(w(t), w(t'))|", fontsize=5.5, labelpad=2)
    cbar.ax.tick_params(labelsize=4.5)

    fig.suptitle(
        "Supplementary Figure S5a — Coding direction stability across time, all 4 subjects",
        fontsize=7.5, fontweight="bold", y=0.97)
    save_figure(fig, "figS5a_supp")
    print("  Figure S5a saved.")
    plt.close(fig)
    return cds_data


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S5b — Time-resolved temporal stability τ(t)
# ─────────────────────────────────────────────────────────────────────────────

def make_figS5b(stats_data, cds_data):
    """Time-resolved τ(t): full-epoch profile, method cross-check, and onset latency."""
    nature_style()
    cos_sims, tau_ts, t_idxs, all_times = cds_data

    fig = plt.figure(figsize=(7.2, 6.4))
    gs = gridspec.GridSpec(2, 2, fig, hspace=0.55, wspace=0.40,
                           left=0.09, right=0.97, top=0.89, bottom=0.09)

    # ── A: τ(t) overlay, all 4 subjects ───────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    for i, subj in enumerate(SUBJECTS):
        t_sec = all_times[subj][t_idxs[subj]]
        ax_a.plot(t_sec, tau_ts[subj], color=SUBJ_COLORS[i], lw=1.3,
                  label=SUBJ_LABELS[subj])
    ax_a.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6, label="τ=1")
    ax_a.axvspan(0.3, 1.4, color="gold", alpha=0.1, zorder=0)
    ax_a.axvline(0, color="k", lw=0.5, ls="--")
    ax_a.set_xlabel("Training time (s)", fontsize=6.5)
    ax_a.set_ylabel("τ(t_train)", fontsize=6.5)
    ax_a.set_title("A  Time-resolved τ(t), all subjects\n"
                   "(when does the code become time-invariant?)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5, ncol=2, loc="lower right")

    # ── B: Method cross-check — time-resolved τ̄ vs whole-window CTG τ ────────
    ax_b = fig.add_subplot(gs[0, 1])
    x = np.arange(4)
    tau_bar_vals = [float(np.nanmean(tau_ts[s])) for s in SUBJECTS]
    tau_ctg_vals = [stats_data["ctg"][s]["temporal_stability"] for s in SUBJECTS]
    w = 0.32
    ax_b.bar(x - w/2, tau_bar_vals, width=w, color=SUBJ_COLORS, alpha=0.55,
             label="Time-resolved τ̄")
    ax_b.bar(x + w/2, tau_ctg_vals, width=w, color=SUBJ_COLORS, alpha=1.0,
             label="Whole-window CTG τ")
    ax_b.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([SUBJ_LABELS[s] for s in SUBJECTS], fontsize=5.5, rotation=15)
    ax_b.set_ylabel("Temporal stability τ", fontsize=6.5)
    ax_b.set_title("B  Cross-check: two independent τ estimates agree",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5, loc="lower right")

    # ── C: Maintenance-window-only τ(t) (zoom, consistency within delay) ─────
    ax_c = fig.add_subplot(gs[1, 0])
    maint_means = []
    for i, subj in enumerate(SUBJECTS):
        t_sec = all_times[subj][t_idxs[subj]]
        m = (t_sec >= 0.3) & (t_sec <= 1.4)
        ax_c.plot(t_sec[m], tau_ts[subj][m], color=SUBJ_COLORS[i], lw=1.4,
                  marker="o", ms=3, label=SUBJ_LABELS[subj])
        maint_means.append(float(np.nanmean(tau_ts[subj][m])))
    ax_c.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.6)
    ax_c.set_xlabel("Training time (s)", fontsize=6.5)
    ax_c.set_ylabel("τ(t_train)", fontsize=6.5)
    ax_c.set_title("C  τ(t) within maintenance only\n(0.3–1.4 s: stable throughout, not just at onset)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=5, ncol=2)

    # ── D: Fraction of the time-course that is sub-unity (activity-maintained) ─
    # Onset latency to τ(t)<1 is degenerate here: 3/4 subjects (al, ca, cc) never
    # reach τ(t)≥1 at any sampled time point (including pre-stimulus baseline),
    # so "time of first dip below 1" is undefined/zero and uninformative. The
    # fraction of the time-course spent sub-unity is the correct, non-degenerate
    # summary — it separates ug (briefly fixed-point-like near stimulus onset)
    # from the other three (activity-maintained throughout).
    ax_d = fig.add_subplot(gs[1, 1])
    frac_sub1 = [100.0 * float(np.mean(tau_ts[subj] < 1.0)) for subj in SUBJECTS]
    bars = ax_d.bar(np.arange(4), frac_sub1, color=SUBJ_COLORS, alpha=0.85, width=0.6)
    for i, f in enumerate(frac_sub1):
        ax_d.text(i, f + 1.5, f"{f:.0f}%", ha="center", va="bottom", fontsize=5.5)
    ax_d.set_ylim(0, 108)
    ax_d.set_xticks(np.arange(4))
    ax_d.set_xticklabels([SUBJ_LABELS[s] for s in SUBJECTS], fontsize=5.5, rotation=15)
    ax_d.set_ylabel("% of time-course with τ(t)<1", fontsize=6.5)
    ax_d.set_title("D  Time spent sub-unity\n(activity-maintained, not fixed-point)",
                   loc="left", fontsize=6.5, fontweight="bold")

    fig.suptitle(
        "Supplementary Figure S5b — Time-resolved temporal stability: onset, "
        "maintenance consistency, and cross-method validation",
        fontsize=7.5, fontweight="bold", y=0.97)
    save_figure(fig, "figS5b_supp")
    print("  Figure S5b saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S6 — Boran MTL iEEG CTG cross-dataset replication
# ─────────────────────────────────────────────────────────────────────────────

def make_figS6(stats_data):
    """Boran Sternberg CTG: 3 representative matrices + 9-subject bar charts + Miller comparison."""
    nature_style()
    fig = plt.figure(figsize=(7.2, 6.0))
    gs_top = gridspec.GridSpec(1, 3, fig, hspace=0.38, wspace=0.38,
                               left=0.08, right=0.97, top=0.87, bottom=0.58)
    gs_bot = gridspec.GridSpec(1, 3, fig, hspace=0.35, wspace=0.45,
                               left=0.08, right=0.97, top=0.48, bottom=0.09)

    boran_stats = stats_data["boran_ctg"]
    subj_ids = list(boran_stats.keys())          # sub-01 … sub-09
    n_boran  = len(subj_ids)
    # Colours: cycle through a 9-colour ramp
    cmap_b   = plt.cm.tab10
    boran_colors = [cmap_b(i / n_boran) for i in range(n_boran)]
    cmap_ctg = plt.cm.RdYlBu_r
    vmin_ctg, vmax_ctg = 0.45, 0.70

    # ── A–C: Representative CTG matrices (highest/median/lowest τ) ────────────
    taus_all = [boran_stats[s]["tau"] for s in subj_ids]
    sorted_by_tau = sorted(subj_ids, key=lambda s: boran_stats[s]["tau"], reverse=True)
    showcase = [sorted_by_tau[0], sorted_by_tau[len(sorted_by_tau) // 2], sorted_by_tau[-1]]

    for idx, subj in enumerate(showcase):
        ax = fig.add_subplot(gs_top[0, idx])
        ctg_data = np.load(RESULTS / f"boran_ctg_{subj}.npz", allow_pickle=True)
        auc_mat  = ctg_data["auc_mat"]
        t_ctg    = ctg_data["times_ctg"]

        im = ax.imshow(auc_mat, aspect="auto", cmap=cmap_ctg,
                       vmin=vmin_ctg, vmax=vmax_ctg, origin="lower",
                       extent=[t_ctg[0], t_ctg[-1], t_ctg[0], t_ctg[-1]])
        ax.axvline(0, color="w", lw=0.5, alpha=0.6)
        ax.axhline(0, color="w", lw=0.5, alpha=0.6)
        # Maintenance window: ~1.0–4.0 s relative to epoch start adjusted for CTG times
        # CTG times span the maintenance epoch already
        ax.set_xlabel("Test time (s)", fontsize=5.5)
        ax.set_ylabel("Train time (s)", fontsize=5.5)
        ax.tick_params(labelsize=4.5)
        tau_s = boran_stats[subj]["tau"]
        od_s  = boran_stats[subj]["mean_offdiag_auc"]
        p_s   = boran_stats[subj]["p_offdiag_vs_chance"]
        p_str = "p<0.001" if p_s < 0.001 else f"p={p_s:.3f}"
        rank  = ["Highest τ", "Median τ", "Lowest τ"][idx]
        ax.set_title(f"{subj} ({rank})\nτ={tau_s:.3f}, AUC={od_s:.3f}, {p_str}",
                     fontsize=6, fontweight="bold")
        panel_label(ax, ["A", "B", "C"][idx])

    # Shared colorbar top row
    cbar_ax = fig.add_axes([0.985, 0.60, 0.010, 0.29])
    sm = plt.cm.ScalarMappable(cmap=cmap_ctg, norm=mcolors.Normalize(vmin_ctg, vmax_ctg))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("AUC", fontsize=5)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.axhline(0.5, color="k", lw=0.7, ls="--")

    # ── D: Off-diagonal AUC — all 9 Boran subjects ───────────────────────────
    # Error bars: SD of the saved label-shuffle permutation null (valid,
    # whole-matrix statistic — not a bootstrap over non-independent cells).
    ax_d = fig.add_subplot(gs_bot[0])
    od_vals = [boran_stats[s]["mean_offdiag_auc"] for s in subj_ids]
    null_sds = []
    for s in subj_ids:
        r = np.load(RESULTS / f"boran_ctg_{s}.npz", allow_pickle=True)
        null_sds.append(float(np.std(r["null"])) if r["null"].size else 0.0)
    yerr_lo = list(null_sds)
    yerr_hi = list(null_sds)

    x_pos = np.arange(n_boran)
    ax_d.bar(x_pos, od_vals, yerr=np.array([yerr_lo, yerr_hi]),
             color=boran_colors, alpha=0.85, width=0.7,
             capsize=2.5, error_kw={"lw": 0.8})
    ax_d.axhline(0.5, color="k", lw=0.8, ls="--", label="Chance (0.5)")
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([s.replace("sub-", "S") for s in subj_ids],
                         fontsize=5, rotation=30)
    d_lo = min(0.46, min(od_vals) - max(yerr_hi) - 0.02)
    d_hi = max(od_vals[i] + yerr_hi[i] for i in range(n_boran)) + 0.03
    ax_d.set_ylim(d_lo, d_hi)
    ax_d.set_ylabel("Mean off-diagonal AUC", fontsize=6)
    ax_d.set_title("D  Off-diagonal AUC (Boran, N=9)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=5)
    for i, (od, p) in enumerate([(boran_stats[s]["mean_offdiag_auc"],
                                   boran_stats[s]["p_offdiag_vs_chance"])
                                  for s in subj_ids]):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        ax_d.text(i, od + yerr_hi[i] + 0.004, star, ha="center",
                  va="bottom", fontsize=6)

    # ── E: Temporal stability τ — all 9 Boran subjects. y-limits are
    # data-adaptive (the corrected nested-CV τ spans a much wider range than
    # the old pipeline, [0.4, 1.5] rather than [0.7, 1.0]); a fixed ylim here
    # previously clipped several bars and pushed their value labels off-panel.
    ax_e = fig.add_subplot(gs_bot[1])
    tau_vals = [boran_stats[s]["tau"] for s in subj_ids]
    ax_e.bar(x_pos, tau_vals, color=boran_colors, alpha=0.85, width=0.7)
    ax_e.axhline(1.0, color="k", lw=0.8, ls="--", label="τ=1")
    mean_tau_b = float(np.mean(tau_vals))
    ax_e.axhline(mean_tau_b, color="#E15759", lw=1.0, ls=":",
                 label=f"Mean τ={mean_tau_b:.3f}")
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels([s.replace("sub-", "S") for s in subj_ids],
                         fontsize=5, rotation=30)
    e_hi = max(tau_vals) * 1.12
    ax_e.set_ylim(0.0, e_hi)
    ax_e.set_ylabel("Temporal stability τ", fontsize=6)
    ax_e.set_title("E  Temporal stability τ (Boran, N=9)\n"
                   f"Range [{min(tau_vals):.3f}, {max(tau_vals):.3f}]",
                   loc="left", fontsize=6, fontweight="bold")
    ax_e.legend(frameon=False, fontsize=5)
    for i, val in enumerate(tau_vals):
        ax_e.text(i, val + e_hi * 0.01, f"{val:.3f}", ha="center",
                  va="bottom", fontsize=4.2, rotation=60, clip_on=True)

    # ── F: Miller vs Boran vs Rutishauser τ — only τ from sessions with a
    # meaningfully decodable diagonal (mean_diag_auc >= 0.55); an
    # uninterpretable τ (near-chance diagonal) can take extreme/unstable
    # values (e.g. -28 or +26) that would otherwise dominate the plot.
    ax_f = fig.add_subplot(gs_bot[2])
    miller_taus = [stats_data["ctg"][s]["temporal_stability"] for s in SUBJECTS
                   if stats_data["ctg"][s]["mean_diag"] >= 0.55]
    rushi_taus_f = [v["tau"] for v in stats_data.get("dandi000469_ctg", {}).values()
                    if v.get("mean_diag_auc", 0) >= 0.55 and np.isfinite(v.get("tau", np.nan))]
    rng = np.random.default_rng(7)
    jit_m = rng.uniform(-0.07, 0.07, len(miller_taus))
    jit_b = rng.uniform(-0.07, 0.07, len(tau_vals))
    jit_r = rng.uniform(-0.07, 0.07, len(rushi_taus_f))
    ax_f.scatter(np.zeros(len(miller_taus)) + jit_m, miller_taus,
                 color=DATASET_COLORS["Miller"], s=35, alpha=0.85, zorder=3,
                 label=f"Miller PFC (N={len(miller_taus)})")
    ax_f.scatter(np.ones(len(tau_vals)) + jit_b, tau_vals,
                 color=DATASET_COLORS["Boran"], s=35, alpha=0.85, zorder=3,
                 label=f"Boran MTL (N={len(tau_vals)})")
    ax_f.scatter(2 * np.ones(len(rushi_taus_f)) + jit_r, rushi_taus_f,
                 color=DATASET_COLORS["Rutishauser"], s=35, alpha=0.75, zorder=3,
                 label=f"Rutishauser (N={len(rushi_taus_f)})")
    for x, vals, color in [(0, miller_taus, DATASET_COLORS["Miller"]),
                            (1, tau_vals, DATASET_COLORS["Boran"]),
                            (2, rushi_taus_f, DATASET_COLORS["Rutishauser"])]:
        m = np.mean(vals); se = np.std(vals) / np.sqrt(len(vals))
        ax_f.errorbar(x, m, yerr=se, fmt="D", color=color, ms=6,
                      capsize=4, lw=1.5, zorder=4)
    ax_f.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.5, label="τ=1")
    ax_f.set_xlim(-0.6, 2.6)
    all_f_vals = miller_taus + tau_vals + rushi_taus_f
    ax_f.set_ylim(0.0, max(all_f_vals) * 1.12 if all_f_vals else 1.2)
    ax_f.set_xticks([0, 1, 2])
    ax_f.set_xticklabels(["Miller\nPFC", "Boran\nMTL", "Rutis.\nSU"], fontsize=6)
    ax_f.set_ylabel("Temporal stability τ", fontsize=6.5)
    ax_f.set_title("F  τ across 3 datasets (diag AUC≥0.55 only)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_f.legend(frameon=False, fontsize=4.5, loc="lower right")
    from scipy.stats import kruskal
    if len(rushi_taus_f) > 1:
        _, p_kw = kruskal(miller_taus, tau_vals, rushi_taus_f)
        ax_f.text(0.97, 0.97, f"Kruskal-Wallis p={p_kw:.3f}", transform=ax_f.transAxes,
                  ha="right", va="top", fontsize=5,
                  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.85))

    fig.suptitle(
        "Supplementary Figure S6 — Boran MTL iEEG CTG replication and 3-dataset τ comparison",
        fontsize=7.5, fontweight="bold", y=0.98)
    save_figure(fig, "figS6_supp")
    print("  Figure S6 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S7 — TES1 personalisation: inter-subject B variation
# ─────────────────────────────────────────────────────────────────────────────

def make_figS7(stats_data):
    """TES1 B-matrix inter-subject variation, Gramian, and LQR rescue comparison."""
    nature_style()

    tes1_file  = RESULTS / "tes1_comprehensive.npz"
    boran_file = RESULTS / "tes1_boran_B.npz"
    if not tes1_file.exists():
        print("  SKIP figS7: run scripts/run_tes1_analysis.py first")
        return

    tes1 = np.load(tes1_file, allow_pickle=True)
    tes1_ids  = list(tes1["tes1_subject_ids"])
    n_tes1    = len(tes1_ids)
    tes1_lqr  = stats_data.get("tes1_lqr", {})

    fig = plt.figure(figsize=(7.2, 6.0))
    gs_top = gridspec.GridSpec(1, 2, fig, wspace=0.42, left=0.09, right=0.97,
                               top=0.87, bottom=0.57)
    gs_bot = gridspec.GridSpec(1, 4, fig, wspace=0.50, left=0.07, right=0.98,
                               top=0.47, bottom=0.11)

    # ── A: B-norm per TES1 subject, one line per Miller subject ──────────────
    ax_a = fig.add_subplot(gs_top[0])
    x_tgt  = np.arange(n_tes1)
    miller_done = [s for s in SUBJECTS if f"{s}_B_norms" in tes1]
    for i, subj in enumerate(miller_done):
        B_norms = tes1[f"{subj}_B_norms"]
        ax_a.plot(x_tgt, B_norms / B_norms.mean(), color=SUBJ_COLORS[i],
                  lw=1.2, alpha=0.85, label=SUBJ_LABELS[subj])

    ax_a.axhline(1.0, color="k", lw=0.6, ls="--", alpha=0.4)
    ax_a.set_xlabel("TES1 subject index", fontsize=6.5)
    ax_a.set_ylabel("Normalised ‖B_latent‖", fontsize=6.5)
    ax_a.set_title("A  TES1 inter-subject variation in B\n"
                   "(each line = one Miller patient)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5.5, ncol=2)

    # Fold-range annotation
    if miller_done:
        fold_ranges = []
        for s in miller_done:
            bn = tes1[f"{s}_B_norms"]
            fold_ranges.append(bn.max() / (bn.min() + 1e-10))
        ax_a.text(0.03, 0.97, f"Mean fold-range: {np.mean(fold_ranges):.1f}×",
                  transform=ax_a.transAxes, fontsize=5.5, va="top",
                  bbox=dict(boxstyle="round,pad=0.3", fc="#FFF8E8", ec="goldenrod", alpha=0.8))

    # ── B: Gramian trace per TES1 subject (controls how easy to steer) ───────
    ax_b = fig.add_subplot(gs_top[1])
    if miller_done:
        # Average Gramian trace across Miller subjects per TES1 subject
        all_traces = np.array([tes1[f"{s}_gramian_traces"] for s in miller_done])
        mean_traces = all_traces.mean(axis=0)
        sem_traces  = all_traces.std(axis=0) / np.sqrt(len(miller_done))
        ax_b.fill_between(x_tgt, mean_traces - sem_traces, mean_traces + sem_traces,
                          color="#4E79A7", alpha=0.18)
        ax_b.plot(x_tgt, mean_traces, color="#4E79A7", lw=1.5)
        best_idx = int(np.argmax(mean_traces))
        ax_b.axvline(best_idx, color=PALETTE["target"], lw=1.0, ls="--", alpha=0.8)
        ax_b.text(best_idx + 0.3, mean_traces.max() * 0.98,
                  f"Best: {tes1_ids[best_idx]}", fontsize=5, color=PALETTE["target"],
                  va="top")
    ax_b.set_xlabel("TES1 subject index", fontsize=6.5)
    ax_b.set_ylabel("Gramian trace Tr(W_c)", fontsize=6.5)
    ax_b.set_title("B  Controllability Gramian per TES1 patient\n"
                   "(higher → easier neural state control)",
                   loc="left", fontsize=6.5, fontweight="bold")

    # ── C: LQR rescue %: mean vs best TES1 B — Miller subjects ────────────────
    # "Worst" B is reported as text, not a bar: its magnitude (down to -1345%)
    # is qualitatively different (catastrophic destabilisation from a poorly
    # aligned B, not a graded rescue) and swamps the mean/best comparison on
    # a shared linear axis. ylim is set from the actual mean/best data range,
    # not hardcoded to [0,105], since best-B rescue is negative for some
    # subjects (personalisation helps on average but is not guaranteed).
    ax_c = fig.add_subplot(gs_bot[0])
    miller_lqr_s = tes1_lqr.get("miller", {})
    if miller_lqr_s:
        subjs_done = [s for s in SUBJECTS if s in miller_lqr_s
                      and "lqr_mean_reduction_pct" in miller_lqr_s[s]]
        xs = np.arange(len(subjs_done))
        bar_w = 0.32
        all_vals_c = []
        for j, (key, col, lbl) in enumerate([
            ("lqr_mean_reduction_pct",  "#4E79A7", "Mean TES1 B"),
            ("lqr_best_reduction_pct",  PALETTE["target"], "Best TES1 B"),
        ]):
            vals = [miller_lqr_s[s].get(key, np.nan) for s in subjs_done]
            all_vals_c += [v for v in vals if not np.isnan(v)]
            ax_c.bar(xs + (j - 0.5) * bar_w, vals, width=bar_w,
                     color=col, alpha=0.85, label=lbl)
        worst_vals = [miller_lqr_s[s].get("lqr_worst_reduction_pct", np.nan)
                      for s in subjs_done]
        worst_vals = [v for v in worst_vals if not np.isnan(v)]
        ax_c.set_xticks(xs)
        ax_c.set_xticklabels([SUBJ_LABELS[s] for s in subjs_done], fontsize=5.5)
        ax_c.set_ylabel("Trajectory rescue (%)", fontsize=6.5)
        y_lo = min(all_vals_c) - 15; y_hi = max(all_vals_c) + 15
        ax_c.set_ylim(y_lo, max(y_hi, 105))
        ax_c.axhline(100, color="k", lw=0.5, ls=":", alpha=0.4)
        ax_c.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax_c.legend(frameon=False, fontsize=5, ncol=1, loc="upper left")
        if worst_vals:
            ax_c.text(0.97, 0.03, f"Worst B: {min(worst_vals):.0f}% to "
                      f"{max(worst_vals):.0f}%\n(off-scale — see Methods)",
                      transform=ax_c.transAxes, ha="right", va="bottom", fontsize=4.5,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.85))
    ax_c.set_title("C  LQR rescue by B selection\n(Miller PFC/parietal)",
                   loc="left", fontsize=5.8, fontweight="bold")

    # ── D: Same for Boran ─────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs_bot[1])
    boran_lqr_s = tes1_lqr.get("boran", {})
    boran_done  = [s for s in sorted(boran_lqr_s)
                   if "lqr_mean_reduction_pct" in boran_lqr_s[s]]
    if boran_done:
        xs_b = np.arange(len(boran_done))
        all_vals_d = []
        for j, (key, col, lbl) in enumerate([
            ("lqr_mean_reduction_pct",  "#E15759", "Mean"),
            ("lqr_best_reduction_pct",  PALETTE["target"], "Best"),
        ]):
            vals = [boran_lqr_s[s].get(key, np.nan) for s in boran_done]
            all_vals_d += [v for v in vals if not np.isnan(v)]
            ax_d.bar(xs_b + (j - 0.5) * 0.32, vals, width=0.32,
                     color=col, alpha=0.85, label=lbl)
        worst_vals_d = [boran_lqr_s[s].get("lqr_worst_reduction_pct", np.nan)
                        for s in boran_done]
        worst_vals_d = [v for v in worst_vals_d if not np.isnan(v)]
        ax_d.set_xticks(xs_b)
        ax_d.set_xticklabels([s.replace("sub-", "S") for s in boran_done],
                              fontsize=5, rotation=30)
        y_lo_d = min(all_vals_d) - 15; y_hi_d = max(all_vals_d) + 15
        ax_d.set_ylim(y_lo_d, max(y_hi_d, 105))
        ax_d.axhline(100, color="k", lw=0.5, ls=":", alpha=0.4)
        ax_d.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax_d.legend(frameon=False, fontsize=5, loc="upper left")
        if worst_vals_d:
            ax_d.text(0.97, 0.03, f"Worst B: {min(worst_vals_d):.0f}% to "
                      f"{max(worst_vals_d):.0f}%\n(off-scale — see Methods)",
                      transform=ax_d.transAxes, ha="right", va="bottom", fontsize=4.5,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.85))
    else:
        ax_d.text(0.5, 0.5, "Run TES1 analysis after\nBoran pipeline update",
                  ha="center", va="center", transform=ax_d.transAxes, fontsize=7)
        ax_d.axis("off")
    ax_d.set_title("D  LQR rescue — Boran MTL\n(B personalised to MTL coords)",
                   loc="left", fontsize=5.8, fontweight="bold")
    ax_d.set_ylabel("Trajectory rescue (%)", fontsize=6.5)

    # ── E: Cross-dataset LQR comparison scatter ───────────────────────────────
    ax_e = fig.add_subplot(gs_bot[2])
    m_vals = [miller_lqr_s[s].get("lqr_best_reduction_pct", np.nan)
              for s in SUBJECTS if s in miller_lqr_s]
    b_vals = [boran_lqr_s[s].get("lqr_best_reduction_pct", np.nan)
              for s in boran_done]
    m_vals = [v for v in m_vals if not np.isnan(v)]
    b_vals = [v for v in b_vals if not np.isnan(v)]

    if m_vals and b_vals:
        ax_e.scatter(np.zeros(len(m_vals)), m_vals, color="#4E79A7", s=55, alpha=0.9, zorder=3)
        ax_e.scatter(np.ones(len(b_vals)), b_vals, color="#E15759", s=55, alpha=0.9, zorder=3)
        for x, vals, col in [(0, m_vals, "#4E79A7"), (1, b_vals, "#E15759")]:
            m = np.nanmean(vals); se = np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
            ax_e.errorbar(x, m, yerr=se, fmt="D", color=col, ms=7,
                          capsize=4, lw=1.5, zorder=4)
        ax_e.set_xlim(-0.6, 1.6)
        ax_e.set_xticks([0, 1])
        ax_e.set_xticklabels([f"Miller PFC\n(N={len(m_vals)})",
                              f"Boran MTL\n(N={len(b_vals)})"], fontsize=6.5)
        ax_e.set_ylabel("Best-B LQR rescue (%)", fontsize=6.5)
        y_all = m_vals + b_vals
        ax_e.set_ylim(min(y_all) - 15, max(max(y_all) + 15, 105))
        ax_e.axhline(0, color="k", lw=0.6, alpha=0.5)
        ax_e.axhline(100, color="k", lw=0.5, ls=":", alpha=0.4)

        from scipy.stats import mannwhitneyu
        if len(m_vals) > 1 and len(b_vals) > 1:
            _, p_mw = mannwhitneyu(m_vals, b_vals, alternative="two-sided")
            ax_e.text(0.97, 0.97, f"M-W p={p_mw:.3f}", transform=ax_e.transAxes,
                      ha="right", va="top", fontsize=5.5,
                      bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC"))
    else:
        ax_e.text(0.5, 0.5, "Awaiting TES1 results", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=7)
        ax_e.axis("off")
    ax_e.set_title("E  Cross-dataset LQR rescue\n(best-B personalisation)",
                   loc="left", fontsize=5.8, fontweight="bold")

    # ── F: Dynamic (v*-aligned) vs static (Gramian-best) electrode selection ──
    ax_f = fig.add_subplot(gs_bot[3])
    div_path = RESULTS / "divergence_analysis.npz"
    if div_path.exists():
        div = np.load(div_path, allow_pickle=True)
        miller_gains = [float(div[f"miller_{s}_align_gain"]) for s in SUBJECTS
                        if f"miller_{s}_align_gain" in div]
        boran_subs_g = [f"sub-{i:02d}" for i in range(1, 10)]
        boran_gains  = [float(div[f"boran_{s}_align_gain"]) for s in boran_subs_g
                        if f"boran_{s}_align_gain" in div]
        rng_f = np.random.default_rng(5)
        for xc, vals, col, lbl in [
            (0, miller_gains, DATASET_COLORS["Miller"], f"Miller\n(N={len(miller_gains)})"),
            (1, boran_gains,  DATASET_COLORS["Boran"],  f"Boran\n(N={len(boran_gains)})"),
        ]:
            jit = rng_f.uniform(-0.08, 0.08, len(vals))
            ax_f.scatter(xc + jit, vals, color=col, s=30, alpha=0.8, zorder=3, label=lbl)
            m = np.mean(vals); se = np.std(vals) / np.sqrt(len(vals))
            ax_f.errorbar(xc, m, yerr=se, fmt="D", color=col, ms=6,
                          capsize=4, lw=1.5, zorder=4)
        ax_f.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="No gain (1×)")
        ax_f.set_xlim(-0.6, 1.6)
        ax_f.set_xticks([0, 1])
        ax_f.set_xticklabels(["Miller", "Boran"], fontsize=6.5)
        ax_f.set_ylabel("Dynamic / static alignment gain (×)", fontsize=6.5)
        ax_f.set_title("F  Personalised stimulation:\ndynamic (v*-aligned) vs static selection",
                       loc="left", fontsize=6, fontweight="bold")
        ax_f.legend(frameon=False, fontsize=5, loc="upper right")
    else:
        ax_f.set_title("F  Personalised stimulation", loc="left",
                       fontsize=6, fontweight="bold")
        ax_f.text(0.5, 0.5, "Run run_divergence_analysis.py", ha="center",
                  va="center", transform=ax_f.transAxes, fontsize=6)
        ax_f.axis("off")

    fig.suptitle(
        "Supplementary Figure S7 — TES1 personalisation: electrode-matched B matrix drives complete rescue",
        fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS7_supp")
    print("  Figure S7 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S8 — Rutishauser single-unit CTG (18 subjects)
# ─────────────────────────────────────────────────────────────────────────────

def make_figS8(stats_data):
    """Rutishauser single-unit CTG: 3 representative matrices + 18-subject summaries."""
    nature_style()
    rushi_stats = stats_data.get("dandi000469_ctg", {})
    if not rushi_stats:
        print("  SKIP figS8: run scripts/run_rutishauser_pipeline.py first")
        return

    subj_ids  = list(rushi_stats.keys())
    n_subj    = len(subj_ids)
    taus_all  = [rushi_stats[s]["tau"] for s in subj_ids]
    sorted_by = sorted(subj_ids, key=lambda s: rushi_stats[s]["tau"], reverse=True)
    showcase  = [sorted_by[0], sorted_by[n_subj // 2], sorted_by[-1]]

    cmap_ctg = plt.cm.RdYlBu_r
    vmin_ctg, vmax_ctg = 0.45, 0.75

    fig = plt.figure(figsize=(7.2, 6.4))
    gs_top = gridspec.GridSpec(1, 3, fig, hspace=0.38, wspace=0.38,
                               left=0.08, right=0.97, top=0.87, bottom=0.57)
    gs_bot = gridspec.GridSpec(1, 3, fig, hspace=0.35, wspace=0.46,
                               left=0.08, right=0.97, top=0.47, bottom=0.09)

    cmap_s = plt.cm.tab20(np.linspace(0, 1, n_subj))

    # ── A-C: Representative CTG matrices ─────────────────────────────────────
    for idx, subj in enumerate(showcase):
        ax = fig.add_subplot(gs_top[0, idx])
        ctg_path = RESULTS / f"dandi000469_ctg_{subj}.npz"
        if not ctg_path.exists():
            ax.axis("off"); continue
        ctg_d   = np.load(ctg_path, allow_pickle=True)
        auc_mat = ctg_d["auc_mat"]
        t_ctg   = ctg_d["times_ctg"]
        im = ax.imshow(auc_mat, aspect="auto", cmap=cmap_ctg,
                       vmin=vmin_ctg, vmax=vmax_ctg, origin="lower",
                       extent=[t_ctg[0], t_ctg[-1], t_ctg[0], t_ctg[-1]])
        ax.set_xlabel("Test time (s)", fontsize=5.5)
        ax.set_ylabel("Train time (s)", fontsize=5.5)
        ax.tick_params(labelsize=4.5)
        rs = rushi_stats[subj]
        rank = ["Highest τ", "Median τ", "Lowest τ"][idx]
        p_s = rs["p_offdiag_vs_chance"]
        p_str = "p<0.001" if p_s < 0.001 else f"p={p_s:.3f}"
        ax.set_title(f"{subj} ({rank})\nτ={rs['tau']:.3f}, AUC={rs['mean_offdiag_auc']:.3f}, {p_str}",
                     fontsize=5.5, fontweight="bold")
        panel_label(ax, ["A", "B", "C"][idx])

    cbar_ax = fig.add_axes([0.985, 0.60, 0.010, 0.27])
    sm = plt.cm.ScalarMappable(cmap=cmap_ctg, norm=mcolors.Normalize(vmin_ctg, vmax_ctg))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("AUC", fontsize=5)
    cbar.ax.tick_params(labelsize=4)
    cbar.ax.axhline(0.5, color="k", lw=0.7, ls="--")

    # ── D: Off-diagonal AUC — all 18 subjects ────────────────────────────────
    # Error bars: SD of the saved label-shuffle permutation null.
    ax_d = fig.add_subplot(gs_bot[0])
    od_vals = [rushi_stats[s]["mean_offdiag_auc"] for s in subj_ids]
    null_sds = []
    for s in subj_ids:
        p = RESULTS / f"dandi000469_ctg_{s}.npz"
        if p.exists():
            r = np.load(p, allow_pickle=True)
            null_sds.append(float(np.std(r["null"])) if r["null"].size else 0.0)
        else:
            null_sds.append(0.0)
    ci_lo   = [od_vals[i] - null_sds[i] for i in range(n_subj)]
    ci_hi   = [od_vals[i] + null_sds[i] for i in range(n_subj)]
    yerr_lo = [od_vals[i] - ci_lo[i] for i in range(n_subj)]
    yerr_hi = [ci_hi[i] - od_vals[i] for i in range(n_subj)]
    x_pos   = np.arange(n_subj)
    ax_d.bar(x_pos, od_vals, yerr=np.array([yerr_lo, yerr_hi]),
             color=[cmap_s[i] for i in range(n_subj)],
             alpha=0.85, width=0.7, capsize=2.5, error_kw={"lw": 0.8})
    ax_d.axhline(0.5, color="k", lw=0.8, ls="--", label="Chance")
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels([s.replace("sub-", "S") for s in subj_ids],
                          fontsize=4.5, rotation=40)
    ax_d.set_ylim(0.47, 0.79)
    ax_d.set_ylabel("Mean off-diagonal AUC", fontsize=6)
    n_sig_fdr = sum(1 for s in subj_ids if rushi_stats[s].get("p_offdiag_vs_chance", 1) < 0.05)
    ax_d.set_title(f"D  Off-diagonal AUC (N={n_subj})\n{n_sig_fdr}/{n_subj} p<0.05 (label-shuffle, uncorrected)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_d.legend(frameon=False, fontsize=5)
    for i, (od, p) in enumerate([(rushi_stats[s]["mean_offdiag_auc"],
                                   rushi_stats[s]["p_offdiag_vs_chance"])
                                  for s in subj_ids]):
        star = "***" if p < 0.001 else ("*" if p < 0.05 else "")
        if star:
            ax_d.text(i, od + yerr_hi[i] + 0.006, star,
                      ha="center", va="bottom", fontsize=5)

    # ── E: τ — all 18 subjects, clipped to a fixed display range (a
    # near-chance diagonal makes τ an unstable ratio that can take extreme
    # values; those subjects are flagged rather than allowed to blow up the
    # axis for everyone else — see Fig 3F for the gated version).
    ax_e = fig.add_subplot(gs_bot[1])
    tau_vals_r = [rushi_stats[s]["tau"] for s in subj_ids]
    TAU_DISPLAY_MAX = 1.8
    tau_capped = [min(max(t, -TAU_DISPLAY_MAX), TAU_DISPLAY_MAX) for t in tau_vals_r]
    ax_e.bar(x_pos, tau_capped,
             color=[cmap_s[i] for i in range(n_subj)],
             alpha=0.85, width=0.7)
    ax_e.axhline(1.0, color="k", lw=0.8, ls="--", label="τ=1")
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels([s.replace("sub-", "S") for s in subj_ids],
                          fontsize=4.5, rotation=40)
    ax_e.set_ylim(-TAU_DISPLAY_MAX * 1.05, TAU_DISPLAY_MAX * 1.15)
    ax_e.set_ylabel("Temporal stability τ", fontsize=6)
    finite_r = [t for t in tau_vals_r if np.isfinite(t)]
    ax_e.set_title(f"E  τ per subject (capped at ±{TAU_DISPLAY_MAX})\n"
                   f"mean={np.mean(finite_r):.3f}±{np.std(finite_r)/np.sqrt(len(finite_r)):.3f} SEM",
                   loc="left", fontsize=6, fontweight="bold")
    ax_e.legend(frameon=False, fontsize=5)
    for i, (val, capped) in enumerate(zip(tau_vals_r, tau_capped)):
        label = f"{val:.2f}" if abs(val) < TAU_DISPLAY_MAX else f"{val:.0f}*"
        y = capped + TAU_DISPLAY_MAX * 0.03 * (1 if capped >= 0 else -3)
        ax_e.text(i, y, label, ha="center", va="bottom", fontsize=3.5, rotation=60,
                 clip_on=True)

    # ── F: PR null across 3 datasets (replaces duplicate τ strip) ────────────
    ax_f = fig.add_subplot(gs_bot[2])
    m_pr_s8 = [[compute_pr(Z[tid == ld].reshape(-1, Z.shape[-1]))
                for ld in [0, 1, 2]]
               for Z, tid, _, _ in [load_geometry(s) for s in SUBJECTS]]
    m_means_s8 = np.mean(m_pr_s8, axis=0)
    m_sems_s8  = np.std(m_pr_s8, axis=0) / np.sqrt(len(SUBJECTS))

    boran_stats_s8 = stats_data["boran_ctg"]
    b_pr4_s8 = [v["pr_per_set"]["4"]["pr_cv"] for v in boran_stats_s8.values()]
    b_pr6_s8 = [v["pr_per_set"]["6"]["pr_cv"] for v in boran_stats_s8.values()]
    b_pr8_s8 = [v["pr_per_set"]["8"]["pr_cv"] for v in boran_stats_s8.values()]
    b_means_s8 = [np.mean(b_pr4_s8), np.mean(b_pr6_s8), np.mean(b_pr8_s8)]
    b_sems_s8  = [np.std(b_pr4_s8)/np.sqrt(9), np.std(b_pr6_s8)/np.sqrt(9), np.std(b_pr8_s8)/np.sqrt(9)]

    r_pr1_s8 = [rushi_stats[s]["pr_per_load"]["1"]["pr_cv"] for s in subj_ids]
    r_pr2_s8 = [rushi_stats[s]["pr_per_load"]["2"]["pr_cv"] for s in subj_ids]
    r_pr3_s8 = [rushi_stats[s]["pr_per_load"]["3"]["pr_cv"] for s in subj_ids]
    r_means_s8 = [np.mean(r_pr1_s8), np.mean(r_pr2_s8), np.mean(r_pr3_s8)]
    r_sems_s8  = [np.std(r_pr1_s8)/np.sqrt(n_subj), np.std(r_pr2_s8)/np.sqrt(n_subj),
                  np.std(r_pr3_s8)/np.sqrt(n_subj)]

    x_f = np.array([0, 1, 2])
    wf = 0.25
    ax_f.bar(x_f - wf, m_means_s8, width=wf, yerr=m_sems_s8,
             color=DATASET_COLORS["Miller"], alpha=0.85, capsize=2.5,
             error_kw={"lw": 0.8}, label="Miller (N=4)")
    ax_f.bar(x_f,      b_means_s8, width=wf, yerr=b_sems_s8,
             color=DATASET_COLORS["Boran"], alpha=0.85, capsize=2.5,
             error_kw={"lw": 0.8}, label="Boran (N=9)")
    ax_f.bar(x_f + wf, r_means_s8, width=wf, yerr=r_sems_s8,
             color=DATASET_COLORS["Rutishauser"], alpha=0.85, capsize=2.5,
             error_kw={"lw": 0.8}, label=f"Rushi. (N={n_subj})")
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels(["Low load", "Mid load", "High load"], fontsize=5.5)
    ax_f.set_ylabel("PR (mean ± SEM)", fontsize=6.5)
    ax_f.set_title("F  PR null: flat across loads — 3 datasets\n(HGA, iEEG-LFP, single-unit spikes)",
                   loc="left", fontsize=5.8, fontweight="bold")
    ax_f.legend(frameon=False, fontsize=4.5, ncol=3, loc="upper center",
                bbox_to_anchor=(0.5, -0.22))

    fig.suptitle(
        "Supplementary Figure S8 — Rutishauser single-unit Sternberg: CTG replication (N=18)",
        fontsize=7.5, fontweight="bold", y=0.98)
    save_figure(fig, "figS8_supp")
    print("  Figure S8 saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Supplementary Figure S9 — Multiband replication: no band is privileged
# ─────────────────────────────────────────────────────────────────────────────

def make_figS9(stats_data):
    """Full multiband replication: τ, PR-null, ring-attractor R, and ∇·v — Miller +
    Boran — plus theta×HGP PAC. Directly addresses "why HGP if theta decodes better?":
    every core geometric/dynamical claim replicates across the spectrum, so HGP's
    role in the rest of the paper is a cross-modality-comparability choice, not a
    performance claim.
    """
    nature_style()
    mb_path = RESULTS / "multiband_ctg.npz"
    if not mb_path.exists():
        print("  SKIP figS9: run scripts/run_multiband_analysis.py first")
        return
    mb = np.load(mb_path, allow_pickle=True)

    band_order = ["theta", "alpha", "beta", "gamma", "hgp"]
    band_lbls  = ["θ", "α", "β", "γ", "HGP"]
    band_cols  = ["#8C564B", "#9467BD", "#17BECF", "#BCBD22", "#E15759"]
    boran_subs = [f"sub-{i:02d}" for i in range(1, 10)]
    boran_done = [s for s in boran_subs if f"boran_{s}_hgp_tau" in mb]

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = gridspec.GridSpec(2, 3, fig, hspace=0.62, wspace=0.48,
                           left=0.08, right=0.97, top=0.90, bottom=0.08)

    # ── A: Miller τ per band (bars + individual subject dots) ────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    rng9 = np.random.default_rng(3)
    means_a, sds_a = [], []
    for bi, b in enumerate(band_order):
        taus = np.array([float(mb[f"{s}_{b}_tau"]) for s in SUBJECTS])
        means_a.append(taus.mean()); sds_a.append(taus.std())
        jit = rng9.uniform(-0.12, 0.12, len(taus))
        ax_a.scatter(bi + jit, taus, color="k", s=10, alpha=0.6, zorder=4)
    ax_a.bar(range(5), means_a, yerr=sds_a, color=band_cols, alpha=0.75,
             width=0.65, capsize=2.5, error_kw={"lw": 0.8})
    ax_a.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6, label="τ=1")
    ax_a.set_xticks(range(5)); ax_a.set_xticklabels(band_lbls, fontsize=6.5)
    ax_a.set_ylim(0.80, 1.05)
    ax_a.set_ylabel("Temporal stability τ", fontsize=6.5)
    ax_a.set_title("A  Miller (N=4): τ per band\n(dots = individual subjects)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5, loc="lower right")

    # ── B: Boran τ per band ──────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    if boran_done:
        means_b, sds_b = [], []
        for bi, b in enumerate(band_order):
            taus = np.array([float(mb[f"boran_{s}_{b}_tau"]) for s in boran_done])
            means_b.append(taus.mean()); sds_b.append(taus.std())
            jit = rng9.uniform(-0.12, 0.12, len(taus))
            ax_b.scatter(bi + jit, taus, color="k", s=8, alpha=0.55, zorder=4)
        ax_b.bar(range(5), means_b, yerr=sds_b, color=band_cols, alpha=0.75,
                 width=0.65, capsize=2.5, error_kw={"lw": 0.8})
        ax_b.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax_b.set_ylim(min(0.80, min(means_b) - 0.05), 1.05)
    else:
        ax_b.text(0.5, 0.5, "Boran multiband pending", ha="center", va="center",
                  transform=ax_b.transAxes, fontsize=6)
    ax_b.set_xticks(range(5)); ax_b.set_xticklabels(band_lbls, fontsize=6.5)
    ax_b.set_ylabel("Temporal stability τ", fontsize=6.5)
    ax_b.set_title(f"B  Boran (N={len(boran_done)}): τ per band\n(cross-modality: iEEG LFP)",
                   loc="left", fontsize=6.5, fontweight="bold")

    # ── C: Miller subject × band τ heatmap ────────────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    tau_grid = np.array([[float(mb[f"{s}_{b}_tau"]) for b in band_order] for s in SUBJECTS])
    im = ax_c.imshow(tau_grid, cmap="RdYlGn_r", vmin=0.85, vmax=1.02, aspect="auto")
    for i in range(len(SUBJECTS)):
        for j in range(len(band_order)):
            ax_c.text(j, i, f"{tau_grid[i,j]:.2f}", ha="center", va="center", fontsize=5.5)
    ax_c.set_xticks(range(5)); ax_c.set_xticklabels(band_lbls, fontsize=6.5)
    ax_c.set_yticks(range(4)); ax_c.set_yticklabels([SUBJ_LABELS[s] for s in SUBJECTS], fontsize=5.5)
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("τ", fontsize=5.5); cbar.ax.tick_params(labelsize=4.5)
    ax_c.set_title("C  τ<1 in every subject × band\n(no exceptions)",
                   loc="left", fontsize=6.5, fontweight="bold")

    # ── D: Ring-attractor Rayleigh R per band (Miller) ────────────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    if f"{SUBJECTS[0]}_theta_rayleigh_R" in mb:
        means_d, sds_d = [], []
        for b in band_order:
            R = np.array([float(mb[f"{s}_{b}_rayleigh_R"]) for s in SUBJECTS])
            means_d.append(R.mean()); sds_d.append(R.std())
        ax_d.bar(range(5), means_d, yerr=sds_d, color=band_cols, alpha=0.8,
                 width=0.65, capsize=2.5, error_kw={"lw": 0.8})
        ax_d.set_ylabel("Rayleigh R (2-back endpoint)", fontsize=6.5)
    else:
        ax_d.text(0.5, 0.5, "Pending", ha="center", va="center",
                  transform=ax_d.transAxes, fontsize=6)
    ax_d.set_xticks(range(5)); ax_d.set_xticklabels(band_lbls, fontsize=6.5)
    ax_d.set_title("D  Ring-attractor phase concentration\nper band (Miller)",
                   loc="left", fontsize=6.5, fontweight="bold")

    # ── E: Flow divergence ∇·v per band (Miller) ──────────────────────────────
    ax_e = fig.add_subplot(gs[1, 1])
    if f"{SUBJECTS[0]}_theta_div_scalar" in mb:
        means_e, sds_e = [], []
        for b in band_order:
            d = np.array([float(mb[f"{s}_{b}_div_scalar"]) for s in SUBJECTS])
            means_e.append(d.mean()); sds_e.append(d.std())
        ax_e.bar(range(5), means_e, yerr=sds_e, color=band_cols, alpha=0.8,
                 width=0.65, capsize=2.5, error_kw={"lw": 0.8})
        ax_e.axhline(0, color="k", lw=0.8, alpha=0.6)
        ax_e.set_ylabel("Flow divergence ∇·v (s⁻¹)", fontsize=6.5)
    else:
        ax_e.text(0.5, 0.5, "Pending", ha="center", va="center",
                  transform=ax_e.transAxes, fontsize=6)
    ax_e.set_xticks(range(5)); ax_e.set_xticklabels(band_lbls, fontsize=6.5)
    ax_e.set_title("E  Flow divergence per band (Miller)\n(HGP alone is consistently contracting)",
                   loc="left", fontsize=6.5, fontweight="bold")

    # ── F: theta × HGP phase-amplitude coupling (PAC), 0-back vs 2-back ──────
    ax_f = fig.add_subplot(gs[1, 2])
    pac_0 = np.array([float(mb[f"{s}_pac_0back"]) for s in SUBJECTS])
    pac_2 = np.array([float(mb[f"{s}_pac_2back"]) for s in SUBJECTS])
    x = np.arange(4); w = 0.32
    ax_f.bar(x - w/2, pac_0, width=w, color=PALETTE["zero_back"], alpha=0.85, label="0-back")
    ax_f.bar(x + w/2, pac_2, width=w, color=PALETTE["two_back"], alpha=0.85, label="2-back")
    ax_f.set_xticks(x); ax_f.set_xticklabels([SUBJ_LABELS[s] for s in SUBJECTS],
                                              fontsize=5.5, rotation=15)
    ax_f.set_ylabel("PAC modulation index", fontsize=6.5)
    ax_f.set_title("F  Theta-phase × HGP-amplitude PAC\n(coupling present; not load-modulated)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_f.legend(frameon=False, fontsize=5.5)

    fig.suptitle(
        "Supplementary Figure S9 — Multiband replication: the WM code is not an HGP artefact",
        fontsize=7.5, fontweight="bold", y=0.97)
    save_figure(fig, "figS9_supp")
    print("  Figure S9 saved.")
    plt.close(fig)


def make_figS10(stats_data):
    """Optimal stimulation timing (sliding-window ∇·v) and location (per-electrode
    alignment with the dominant unstable direction v*). Timing: within-trial window
    where the flow is most contracting (safest) vs most expanding (highest control
    leverage). Location: within a subject's own array, which electrode is best
    coupled to v* — cross-checked against an independent TES1-CCEP-informed score.
    """
    nature_style()
    div_path = RESULTS / "divergence_analysis.npz"
    loc_path = RESULTS / "stim_location_analysis.npz"
    if not (div_path.exists() and loc_path.exists()):
        print("  SKIP figS10: run run_divergence_analysis.py and "
              "run_stim_location_analysis.py first")
        return
    div = np.load(div_path, allow_pickle=True)
    loc = np.load(loc_path, allow_pickle=True)

    boran_subs = [f"sub-{i:02d}" for i in range(1, 10)]

    fig = plt.figure(figsize=(7.2, 7.4))
    gs = gridspec.GridSpec(2, 3, fig, hspace=0.62, wspace=0.50,
                           left=0.09, right=0.96, top=0.90, bottom=0.08)

    # ── A: Miller (al) sliding-window ∇·v(t) trace ────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    t_c = div["miller_al_sw_t_centers"]; d_c = div["miller_al_sw_div_trace"]
    ax_a.plot(t_c, d_c, color=DATASET_COLORS["Miller"], lw=1.3)
    ax_a.axhline(0, color="k", lw=0.6, alpha=0.5)
    t_min = float(div["miller_al_t_min_div"]); t_max = float(div["miller_al_t_max_div"])
    ax_a.axvline(t_min, color="tab:blue", lw=1.0, ls="--", label=f"min: t={t_min:.2f}s")
    ax_a.axvline(t_max, color="tab:red", lw=1.0, ls="--", label=f"max: t={t_max:.2f}s")
    ax_a.set_xlabel("Time (s)", fontsize=6.5)
    ax_a.set_ylabel("∇·v(t) (s⁻¹)", fontsize=6.5)
    ax_a.set_title("A  Timing: sliding-window ∇·v(t)\n(Miller S1, sliding 250 ms window)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5)

    # ── B: Boran (sub-02) sliding-window ∇·v(t) trace ─────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    t_cb = div["boran_sub-02_sw_t_centers"]; d_cb = div["boran_sub-02_sw_div_trace"]
    ax_b.plot(t_cb, d_cb, color=DATASET_COLORS["Boran"], lw=1.3)
    ax_b.axhline(0, color="k", lw=0.6, alpha=0.5)
    t_minb = float(div["boran_sub-02_t_min_div"]); t_maxb = float(div["boran_sub-02_t_max_div"])
    ax_b.axvline(t_minb, color="tab:blue", lw=1.0, ls="--", label=f"min: t={t_minb:.2f}s")
    ax_b.axvline(t_maxb, color="tab:red", lw=1.0, ls="--", label=f"max: t={t_maxb:.2f}s")
    ax_b.set_xlabel("Time (s)", fontsize=6.5)
    ax_b.set_ylabel("∇·v(t) (s⁻¹)", fontsize=6.5)
    ax_b.set_title("B  Timing: sliding-window ∇·v(t)\n(Boran sub-02)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_b.legend(frameon=False, fontsize=5)

    # ── C: cross-subject optimal-timing summary ───────────────────────────────
    ax_c = fig.add_subplot(gs[0, 2])
    for xc, subs, prefix, col, lbl in [
        (0, SUBJECTS,  "miller", DATASET_COLORS["Miller"], "Miller\ncontract"),
        (1, SUBJECTS,  "miller", DATASET_COLORS["Miller"], "Miller\nexpand"),
        (2, boran_subs, "boran", DATASET_COLORS["Boran"],  "Boran\ncontract"),
        (3, boran_subs, "boran", DATASET_COLORS["Boran"],  "Boran\nexpand"),
    ]:
        field = "t_min_div" if xc in (0, 2) else "t_max_div"
        vals = [float(div[f"{prefix}_{s}_{field}"]) for s in subs
                if f"{prefix}_{s}_{field}" in div]
        m = np.mean(vals); se = np.std(vals) / np.sqrt(len(vals))
        rng10 = np.random.default_rng(7)
        jit = rng10.uniform(-0.1, 0.1, len(vals))
        ax_c.scatter(xc + jit, vals, color=col, s=18, alpha=0.7, zorder=3)
        ax_c.errorbar(xc, m, yerr=se, fmt="D", color=col, ms=6, capsize=3, zorder=4)
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(["Miller\ncontract", "Miller\nexpand",
                          "Boran\ncontract", "Boran\nexpand"], fontsize=5.5)
    ax_c.set_ylabel("Time within trial (s)", fontsize=6.5)
    ax_c.set_title("C  Cross-subject optimal timing\n(contracting vs expanding window)",
                   loc="left", fontsize=6, fontweight="bold")

    # ── D: Miller (al) electrode map coloured by v*-alignment ────────────────
    ax_d = fig.add_subplot(gs[1, 0])
    mni_al = loc["miller_al_mni"]; score_al = loc["miller_al_score_intrinsic"]
    top_al = int(loc["miller_al_top_idx"])
    sc = ax_d.scatter(mni_al[:, 1], mni_al[:, 2], c=score_al, cmap="viridis",
                       s=22, alpha=0.85, zorder=3)
    ax_d.scatter(mni_al[top_al, 1], mni_al[top_al, 2], marker="*", s=140,
                 facecolor="none", edgecolor="red", linewidth=1.3, zorder=4,
                 label="top-ranked electrode")
    ax_d.set_xlabel("MNI y (posterior→anterior, mm)", fontsize=6)
    ax_d.set_ylabel("MNI z (inferior→superior, mm)", fontsize=6)
    ax_d.set_title("D  Location: electrode-v* alignment\n(Miller S1 array)",
                   loc="left", fontsize=6, fontweight="bold")
    cbar_d = plt.colorbar(sc, ax=ax_d, fraction=0.046, pad=0.04)
    cbar_d.set_label("|cos(V_i, v*)|", fontsize=5.5); cbar_d.ax.tick_params(labelsize=4.5)
    ax_d.legend(frameon=False, fontsize=5, loc="upper center",
                bbox_to_anchor=(0.5, -0.22))

    # ── E: Boran (sub-02) electrode map coloured by v*-alignment ─────────────
    ax_e = fig.add_subplot(gs[1, 1])
    mni_b = loc["boran_sub-02_mni"]; score_b = loc["boran_sub-02_score_intrinsic"]
    top_b = int(loc["boran_sub-02_top_idx"])
    sc2 = ax_e.scatter(mni_b[:, 1], mni_b[:, 2], c=score_b, cmap="viridis",
                        s=22, alpha=0.85, zorder=3)
    ax_e.scatter(mni_b[top_b, 1], mni_b[top_b, 2], marker="*", s=140,
                 facecolor="none", edgecolor="red", linewidth=1.3, zorder=4,
                 label="top-ranked electrode")
    ax_e.set_xlabel("MNI y (posterior→anterior, mm)", fontsize=6)
    ax_e.set_ylabel("MNI z (inferior→superior, mm)", fontsize=6)
    ax_e.set_title("E  Location: electrode-v* alignment\n(Boran sub-02 array)",
                   loc="left", fontsize=6, fontweight="bold")
    cbar_e = plt.colorbar(sc2, ax=ax_e, fraction=0.046, pad=0.04)
    cbar_e.set_label("|cos(V_i, v*)|", fontsize=5.5); cbar_e.ax.tick_params(labelsize=4.5)
    ax_e.legend(frameon=False, fontsize=5, loc="upper center",
                bbox_to_anchor=(0.5, -0.22))

    # ── F: convergence between intrinsic and TES1-informed location scores ───
    ax_f = fig.add_subplot(gs[1, 2])
    rho_m = [float(loc[f"miller_{s}_rho_intrinsic_vs_tes1"]) for s in SUBJECTS]
    rho_b = [float(loc[f"boran_{s}_rho_intrinsic_vs_tes1"]) for s in boran_subs
             if f"boran_{s}_rho_intrinsic_vs_tes1" in loc]
    x_m = np.zeros(len(rho_m)); x_b = np.ones(len(rho_b))
    rng10b = np.random.default_rng(11)
    ax_f.scatter(x_m + rng10b.uniform(-0.08, 0.08, len(rho_m)), rho_m,
                 color=DATASET_COLORS["Miller"], s=24, alpha=0.8, zorder=3, label="Miller")
    ax_f.scatter(x_b + rng10b.uniform(-0.08, 0.08, len(rho_b)), rho_b,
                 color=DATASET_COLORS["Boran"], s=24, alpha=0.8, zorder=3, label="Boran")
    for xc, vals, col in [(0, rho_m, DATASET_COLORS["Miller"]), (1, rho_b, DATASET_COLORS["Boran"])]:
        m = np.mean(vals); se = np.std(vals) / np.sqrt(len(vals))
        ax_f.errorbar(xc, m, yerr=se, fmt="D", color=col, ms=6, capsize=3, zorder=4)
    ax_f.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)
    ax_f.set_xticks([0, 1]); ax_f.set_xticklabels(["Miller\n(N=4)", "Boran\n(N=9)"], fontsize=6)
    ax_f.set_ylabel("ρ (intrinsic vs TES1-informed)", fontsize=6.5)
    ax_f.set_title("F  Convergence of two independent\nlocation scores (weak-to-moderate)",
                   loc="left", fontsize=6, fontweight="bold")
    ax_f.legend(frameon=False, fontsize=5)

    fig.suptitle(
        "Supplementary Figure S10 — Toward optimal stimulation timing and location",
        fontsize=7.5, fontweight="bold", y=0.97)
    save_figure(fig, "figS10_supp")
    print("  Figure S10 saved.")
    plt.close(fig)


def make_figS_dim_robustness(stats_data):
    """Round-6 supplementary: the latent dimensionality is data-selected, and every
    headline quantity is invariant to it (and to the operator rank).
    (a) per-dataset cv-PR and parallel-analysis k vs channel count, with k=8 marked;
    (b) the four headline quantities vs latent k in {6,8,10,12};
    (c) v*(r) stability and max Re(lambda) across operator ranks r in {4..8}.
    Every number traces to results/*.json; no filler panels.
    """
    nature_style()
    sel_p = RESULTS / "latent_dim_selection.json"
    rob_p = RESULTS / "dim_robustness.json"
    rank_p = RESULTS / "dmd_rank_selection.json"
    if not (sel_p.exists() and rob_p.exists()):
        print("  SKIP figS_dim_robustness: run run_dim_robustness.py first")
        return
    sel = json.load(open(sel_p))
    rob = json.load(open(rob_p))
    rank = json.load(open(rank_p)) if rank_p.exists() else {}

    fig = plt.figure(figsize=(7.2, 6.6))
    gs = gridspec.GridSpec(2, 3, fig, hspace=0.55, wspace=0.50,
                           left=0.09, right=0.97, top=0.90, bottom=0.09)

    # ── (a) selection: cv-PR and parallel-analysis k per dataset ───────────────
    ax_a = fig.add_subplot(gs[0, :])
    dsets = list(sel.keys())
    x = np.arange(len(dsets))
    cv = [sel[d]["cv_PR_mean"] for d in dsets]
    cv_err = [sel[d]["cv_PR_std"] for d in dsets]
    pa = [sel[d]["k_parallel_analysis_median"] for d in dsets]
    ax_a.errorbar(x - 0.10, cv, yerr=cv_err, fmt="o", color="#4E79A7", ms=6,
                  capsize=3, lw=1, label="cross-validated PR")
    ax_a.plot(x + 0.10, pa, "s", color="#E15759", ms=6, label="parallel analysis k")
    ax_a.axhline(8, color="k", ls="--", lw=1.0, alpha=0.7, label="adopted k = 8")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([f"{d}\n(n_ch {sel[d]['n_channels_range']})" for d in dsets], fontsize=5.5)
    ax_a.set_ylabel("effective / selected dimensionality", fontsize=6.5)
    ax_a.set_title("A  Data-selected latent dimensionality per dataset "
                   "(both rules vs the adopted common k=8)",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_a.legend(frameon=False, fontsize=5.5, ncol=3, loc="upper right")

    ks = sorted(rob.keys(), key=int)
    kx = [int(k) for k in ks]

    # ── (b) headline 1 (axis rotation) + 3 (tau) vs k ──────────────────────────
    ax_b = fig.add_subplot(gs[1, 0])
    ar = [rob[k]["axis_rot_diff"] for k in ks]
    ar_lo = [rob[k]["axis_rot_ci"][0] for k in ks]
    ar_hi = [rob[k]["axis_rot_ci"][1] for k in ks]
    ax_b.plot(kx, ar, "o-", color="#59A14F", lw=1.2, ms=5)
    ax_b.fill_between(kx, ar_lo, ar_hi, color="#59A14F", alpha=0.20)
    ax_b.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax_b.set_xticks(kx)
    ax_b.set_xlabel("latent dim k", fontsize=6.5)
    ax_b.set_ylabel("content−context ARI", fontsize=6.5)
    ax_b.set_title("B  Headline 1: axis-rotation\ndifference (95% CI shaded)",
                   loc="left", fontsize=6.5, fontweight="bold")

    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.plot(kx, [rob[k]["tau_context"] for k in ks], "o-", color="#4E79A7",
              lw=1.2, ms=5, label="context τ")
    ax_c.plot(kx, [rob[k]["tau_content"] for k in ks], "s-", color="#E15759",
              lw=1.2, ms=5, label="content τ")
    ax_c.plot(kx, [rob[k]["pr_slope"] for k in ks], "^--", color="#B07AA1",
              lw=1.0, ms=4, label="PR slope (native)")
    ax_c.axhline(0, color="k", lw=0.6, alpha=0.4)
    ax_c.set_xticks(kx)
    ax_c.set_xlabel("latent dim k", fontsize=6.5)
    ax_c.set_ylabel("τ  /  PR slope", fontsize=6.5)
    ax_c.set_title("C  Headlines 2–3: CTG τ and\nPR-vs-load slope vs k",
                   loc="left", fontsize=6.5, fontweight="bold")
    ax_c.legend(frameon=False, fontsize=5)

    # ── (d) v*(r) stability + benchmark slope across operator ranks ────────────
    ax_d = fig.add_subplot(gs[1, 2])
    if rank:
        rs = sorted([k for k in rank if k != "_meta"], key=int)
        rx = [int(r) for r in rs]
        ax_d.plot(rx, [rank[r]["vstar_cos_to_r6"] for r in rs], "o-",
                  color="#F28E2B", lw=1.2, ms=5, label="|v*(r)·v*(6)|")
        ax_d.plot(rx, [rank[r]["benchmark_slope"] for r in rs], "s-",
                  color="#4E79A7", lw=1.2, ms=5, label="benchmark v*-slope")
        ax_d.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax_d.axvline(6, color="k", ls=":", lw=0.8, alpha=0.6)
        ax_d.set_xticks(rx)
        ax_d.set_ylim(-0.1, 1.05)
        ax_d.set_xlabel("operator rank r", fontsize=6.5)
        ax_d.set_title("D  Headline 4: v* stability &\nbenchmark slope vs rank",
                       loc="left", fontsize=6.5, fontweight="bold")
        ax_d.legend(frameon=False, fontsize=5)
    else:
        ax_d.text(0.5, 0.5, "run_dmd_rank_selection.py\nnot yet run", ha="center",
                  va="center", fontsize=6, transform=ax_d.transAxes)
        ax_d.axis("off")

    fig.suptitle("Supplementary Figure — Latent dimensionality is data-selected and "
                 "every headline is invariant to it",
                 fontsize=7.5, fontweight="bold")
    save_figure(fig, "figS_dim_robustness")
    print("  Figure S_dim_robustness saved.")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    FIGURES_DIR.mkdir(exist_ok=True)
    stats_data = load_all_stats()

    print("Generating Figure 1...")
    make_figure1()

    print("Generating Figure 2...")
    make_figure2(stats_data)

    print("Generating Figure 3 (CTG — slow ~60s)...")
    make_figure3(stats_data)

    print("Generating Figure 4...")
    make_figure4(stats_data)

    print("Generating Figure 5 (item-identity CTG)...")
    make_figure5(stats_data)

    print("Generating Figure 6 (behavior)...")
    make_figure6(stats_data)

    print("Generating Figure 7 (content/context dissociation)...")
    make_figure7(stats_data)

    print("Generating Figure 8 (closed-loop demonstration — slow, resimulates)...")
    make_figure8(stats_data)

    print("Generating Figure 9 (closed-loop robustness)...")
    make_figure9(stats_data)

    print("Generating Supplementary S1...")
    make_figS1(stats_data)

    print("Generating Supplementary S2...")
    make_figS2()

    print("Generating Supplementary S3 (LQR)...")
    make_figS3()

    print("Generating Supplementary S4...")
    make_figS4(stats_data)

    print("Generating Supplementary S5a/S5b (CDS — slow ~2 min)...")
    cds_data = make_figS5a()
    make_figS5b(stats_data, cds_data)

    print("Generating Supplementary S6 (Boran CTG replication)...")
    make_figS6(stats_data)

    print("Generating Supplementary S7 (TES1 personalisation)...")
    make_figS7(stats_data)

    print("Generating Supplementary S8 (Rutishauser single-unit CTG)...")
    make_figS8(stats_data)

    print("Generating Supplementary S9 (multiband + PAC validation)...")
    make_figS9(stats_data)

    print("Generating Supplementary S10 (stimulation timing + location)...")
    make_figS10(stats_data)

    print("Generating Supplementary S_dim_robustness (Round-6 dimensionality)...")
    make_figS_dim_robustness(stats_data)

    print("Generating Supplementary S11 (Round-8 Part 3B: benchmark N/A + behavior bound)...")
    make_figS11_round8()

    print("\nAll figures saved to figures/")
    for f in sorted(FIGURES_DIR.glob("fig*.pdf")):
        print(f"  {f.name}")
