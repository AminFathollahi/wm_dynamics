# WM Dynamics — Manifold Geometry of Working Memory Failure
### Anchor Project: NeuroAI × BCI × Organoid Intelligence

**Question:** Does the geometry of prefrontal population dynamics predict working memory failure, and can we compute the minimum-energy control signal that would prevent it?

**Data:** Kai Miller N-back iEEG (4 subjects, 40-64 ECoG channels, 1200 Hz, 0/1/2-back conditions)

**Pipeline:** iEEG preprocessing → high-gamma power → Sequential VAE → principal angles + tangling → DMD system ID → LQR rescue

---

## Quick Start

```bash
conda env create -f environment.yml
conda activate wm_dynamics

# Symlink the data
ln -sf "../dynamical systems/kai miller/memory_nback/memory_nback" data/miller_nback

# Start here
jupyter lab notebooks/
```

**Start with:** `notebooks/01_data_signal/01_load_explore_ieeg.ipynb`  
**Math foundations:** `notebooks/00_foundations/00a_linear_algebra_eigensystems.ipynb`

---

## Structure

```
wm_dynamics/
├── notebooks/
│   ├── 00_foundations/
│   │   └── 00a_linear_algebra_eigensystems.ipynb   ← START HERE (math)
│   ├── 01_data_signal/
│   │   └── 01_load_explore_ieeg.ipynb              ← START HERE (data)
│   ├── 02_latent_trajectories/
│   │   └── 02_sequential_vae.ipynb
│   ├── 03_geometric_biomarker/
│   │   └── 03_principal_angles_tangling.ipynb
│   ├── 04_system_id/
│   │   └── 04_dmd_and_linear_dynamics.ipynb
│   ├── 05_lqr_control/
│   │   └── 05_lqr_rescue.ipynb
│   └── 06_full_pipeline/
│       └── 06_figures_and_validation.ipynb
├── scripts/
│   └── ieeg_utils.py            ← shared preprocessing + analysis functions
├── notes/
│   └── TEMPLATE.md              ← write one note per new concept (non-negotiable)
├── figures/                     ← output figures go here
├── data/                        ← symlink to Miller data goes here
└── environment.yml
```

---

## Outputs (Publication Figures)

| Figure | Description | Module |
|--------|-------------|--------|
| `fig1_latent_trajectories.pdf` | 3D PCA of latent trajectories, by condition | 02, 06 |
| `fig2_geometric_biomarkers.pdf` | θ_min(t) + Q(t), correct vs incorrect | 03, 06 |
| `fig3_lqr_rescue.pdf` | LQR rescue simulation + energy-accuracy curve | 05, 06 |

---

## Conceptual Contribution

Trajectory tangling and principal angle analysis have been applied to **motor cortex in non-human primates** (Russo et al. 2018). This project applies both frameworks to **human prefrontal cortex during cognitive maintenance** and connects the geometric analysis directly to a **control-theoretic rescue design**. The LQR formulation is the mathematical precursor to a closed-loop BCI stimulation protocol.

---

## Read Along

- Russo et al. (2018) Neuron — the paper this project extends
- Cunningham & Yu (2014) Nat Neurosci — conceptual foundation
- Kingma & Welling (2013) — VAE (before Module 2)
- Stengel — Optimal Control Ch. 4 (before Module 5)
- Strogatz — Nonlinear Dynamics, Ch. 1-6 (before anything)
