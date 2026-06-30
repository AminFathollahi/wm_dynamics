# WM Dynamics — Neural Manifold Geometry and Control in Human Working Memory

**Status:** Pipeline executed | 165/165 tests passing | Results: notebooks 01–06 complete

## Scientific Question

Does the intrinsic geometry of prefrontal high-gamma population dynamics predict working memory (WM) maintenance success and target detection, and can linear control theory guide closed-loop neuromodulation to rescue failing memory circuits?

**Three-part contribution:**
1. **Geometric biomarkers:** Participation ratio (PR) and subspace principal angles as load-dependent manifold descriptors.
2. **Dynamical fingerprints:** Trajectory tangling Q(t) and DMD eigenspectrum characterise maintenance dynamics.
3. **Control-theoretic translation:** LQR-based minimum-energy rescue trajectories with biophysically-grounded input mapping via TES1.

## Preliminary Results (N=4 subjects, Miller N-back ECoG)

| Analysis | Key result | Interpretation |
|----------|-----------|----------------|
| **Preprocessing** | 3 ch rejected (al: 3/40, cc: 3/64, ug: 12/62, ca: 1/64) | MAD-based rejection; subject ug has highest artefact rate |
| **PR vs load** | cc: 4.80→5.18→5.44 ✓; al: non-monotonic; ca/ug: decreasing | Heterogeneous — load effect is subject-specific, not universal |
| **8-PC variance** | al: 38.9%, ca: 60.3%, cc: 44.2%, ug: 69.5% | Wide subject variability in manifold compactness |
| **LOSO AUROC** | 0.603 (p=0.024, permutation) | Geometry features predict target detection above chance (modest effect) |
| **Tangling Q** | Non-target > target (7108 vs 5180) | Opposite to H2b — target trials have lower tangling (more stable dynamics) |
| **DMD \|λ\|** | Target: 0.9998, Non-target: 0.9999 | Both near unit circle; tiny difference between conditions |
| **LQR closed-loop** | E=57.1, final error=1.74 in 150 steps | Feasibility demonstrated; 150-step convergence with 2-channel random B |

**Scientific note on PR heterogeneity:** The mixed PR-load relationship (cc supports H1a; ca and ug show decrease) likely reflects individual differences in electrode coverage and PFC subregion sampling. Subjects ca and ug have high baseline dimensionality (PR ≈ 4), leaving less room for load-dependent expansion. Linear mixed-effects analysis (controlling for subject) is needed for inference.

## Datasets

All data on external USB: `/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/`

| Dataset | N | Signal | Task | Path |
|---------|---|--------|------|------|
| **Miller N-back** | 4 subjects | ECoG 1200 Hz, PFC/parietal | 0/1/2-back verbal N-back | `kai miller/memory_nback/memory_nback/data/{subj}_nback.mat` |
| **Boran Sternberg** (DANDI 000574) | 9 subjects | iEEG 1398 Hz + EEG + single units | Sternberg set sizes 4/6/8 | `000574/sub-{01-09}/` |
| **TES1** (Huang et al. 2017 eLife) | 17 subjects | tES-induced voltages | Stimulation mapping 1 mA | `Tes1/HuangLiu2016dataset.zip` |

## Repository Structure

```
wm_dynamics/
├── src/                       ← Production library (git-tracked)
│   ├── preprocessing.py       ← iEEG: CAR, notch, HGP, epoching (Miller + Boran + TES1)
│   ├── geometry.py            ← PCA, PR, principal angles, CTG, RSA, geometric_drift
│   ├── dynamics.py            ← Tangling Q(t), exact DMD, ring attractor phase, local Jacobian
│   ├── control.py             ← Controllability, LQR/DARE, minimum-energy control
│   ├── statistics.py          ← Bootstrap, cluster permutation, AUROC, LOSO, hedges_g, LME
│   └── visualization.py       ← Nature Neuroscience–style figure utilities
│
├── notebooks/                 ← Production analysis pipeline (git-tracked)
│   ├── 01_preprocessing.ipynb       ← HGP, epoching, QC  [EXECUTED — results/01_epochs_*.npz]
│   ├── 02_latent_geometry.ipynb     ← PCA, PR, principal angles  [EXECUTED — results/02_geometry_*.npz]
│   ├── 03_tangling_dynamics.ipynb   ← Q(t) tangling, DMD  [EXECUTED — results/03_dynamics.npz]
│   ├── 04_predictive_model.ipynb    ← LOSO AUROC, electrode capacity  [EXECUTED — results/04_*.npz]
│   ├── 05_control_theory.ipynb      ← LQR rescue, TES1 B matrix  [EXECUTED — results/05_*.npz]
│   ├── 06_paper_figures.ipynb       ← Figure assembly  [EXECUTED — figures/]
│   ├── 07_rsa_neuroai.ipynb         ← Cross-subject RSA, CKA  [scaffold]
│   ├── 08_extended_analysis.ipynb   ← Ring attractor, CTG, LOSO transfer  [scaffold]
│   └── 09_cross_dataset_replication.ipynb  ← Boran replication, TES1  [scaffold]
│
├── tests/                     ← Unit tests (165/165 passing)
│
├── results/                   ← Saved .npz files (gitignored)
├── figures/                   ← Saved PDF/PNG figures (gitignored)
│
├── learning/                  ← Pedagogical notebooks (gitignored — local only)
│   ├── CURRICULUM.md          ← Full reading list and learning roadmap
│   ├── 00_math_foundations/   ← Linear algebra, ODEs, probability, info theory
│   ├── 01_neural_signals/     ← iEEG/LFP, spectral analysis, spiking
│   ├── 02_dimensionality_reduction/ ← PCA/SVD, autoencoders/VAEs, RSA
│   ├── 03_dynamical_systems/  ← Linear dynamics, DMD, LFADS, ring attractors
│   ├── 04_optimal_control/    ← LQR/DARE, closed-loop BCI
│   ├── 05_statistics/         ← GLMs, permutation/bootstrap, Bayesian
│   ├── 06_deep_learning/      ← MLP, CNN, RNN, transformers, CEBRA
│   └── 07_neuroai/            ← RSA alignment, ANN-brain, normative models
│
├── PAPER_DRAFT.md             ← Scientific analysis plan, hypotheses, methods
├── README.md                  ← This file
└── environment.yml            ← Conda environment (activate: conda activate nmap)
```

## Quickstart

```bash
conda activate nmap

# Run tests
cd /home/amin/Research/Representation/Working\ Memory/wm_dynamics
python -m pytest tests/ -v   # 165/165 passing

# Execute pipeline (requires external USB data for notebook 01)
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  01_preprocessing.ipynb    # → results/01_epochs_*.npz
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=600 \
  02_latent_geometry.ipynb  # → results/02_geometry_*.npz
# ... repeat for 03–06
```

## Methods Summary

### Signal Processing
- Common average reference (CAR) → notch filter (60, 120, 180, 240 Hz) → high-gamma power (70–150 Hz Butterworth → Hilbert envelope² → 50 ms Gaussian σ)
- Epochs: −200 ms to +1500 ms around stimulus onset; baseline z-score per electrode over [−200, 0 ms]
- Channel rejection: MAD-based (threshold 3 MAD units; 1.4826 scaling factor)

### Dimensionality Reduction
- PCA via full SVD on all-trials-pooled maintenance window (300–1400 ms post-onset)
- Participation ratio: PR = (Σλᵢ)² / Σλᵢ² — intrinsic dimensionality
- Principal angles between condition subspaces (Björck & Golub 1973)
- Statistical test: within-subject permutation + linear mixed effects (subject as random intercept)

### Trajectory Dynamics
- Trajectory tangling: Q(t) = max_{t'} ‖Ż(t)−Ż(t')‖² / (‖Z(t)−Z(t')‖² + ε), ε=1e-3 (Russo et al. 2018)
- Vectorised implementation (O(T²) via broadcasting, fallback loop for T > 2000)
- Exact DMD: Tu et al. (2014) algorithm on maintenance window; truncation rank r=6
- Ring attractor phase: top-2 PCA projection + arctan2

### Control Theory
- DARE solution via `scipy.linalg.solve_discrete_are`
- LQR feedback gain K: Q = q·I, R = r·I; Pareto sweep q ∈ [0.01, 100]
- Minimum-energy control via controllability Gramian inversion (regularised by 1e-10·I)
- TES1 anatomical B matrix: Gaussian spatial interpolation from MNI coordinates (σ=20 mm)
- Power estimate: P[µW] = I²[µA²] × Z[kΩ] × 1e-3

### Statistics
- Temporal cluster permutation test (Maris & Oostenveld 2007) — FWER control
- Bootstrap CIs: percentile method, B=5000
- LOSO cross-validation (leave-one-subject-out); AUROC via trapezoidal rule
- Effect sizes: Cohen's d and Hedge's g (bias-corrected for small samples)
- Linear mixed effects: within-subject permutation preserving subject structure

## Key References

1. Cunningham JP & Yu BM (2014) *Dimensionality reduction for large-scale neural recordings.* **Nat Neurosci** 17:1500.
2. Russo AA et al. (2018) *Motor cortex embeds muscle-like commands in an untangled population response.* **Neuron** 97:953.
3. Tu JH et al. (2014) *On dynamic mode decomposition.* **J Comput Dyn** 1(2).
4. Gu S et al. (2015) *Controllability of structural brain networks.* **Nat Commun** 6:8414.
5. Maris E & Oostenveld R (2007) *Nonparametric statistical testing of EEG- and MEG-data.* **J Neurosci Meth** 164:177.
6. Miller KJ et al. (2016) *Spontaneous Decoding of the Timing and Content of Human Object Perception.* **PLoS Biol** 14(2).
7. Panichello MF & Buschman TJ (2021) *Shared mechanisms underlie the control of working memory and attention.* **Nature** 592:601.
8. Huang Y et al. (2017) *Measurements and models of electric fields during tES.* **eLife** 6:e18834.
