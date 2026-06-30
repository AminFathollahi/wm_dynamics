# WM Dynamics — Neural Manifold Geometry and Control in Human Working Memory

**Status:** Pipeline executed | 165/165 tests passing | Results: notebooks 01–09 complete

## Scientific Question

Does the intrinsic geometry of prefrontal high-gamma population dynamics predict working memory (WM) maintenance success and target detection, and can linear control theory guide closed-loop neuromodulation to rescue failing memory circuits?

**Three-part contribution:**
1. **Geometric biomarkers:** Participation ratio (PR) and subspace principal angles as load-dependent manifold descriptors.
2. **Dynamical fingerprints:** Trajectory tangling Q(t) and DMD eigenspectrum characterise maintenance dynamics.
3. **Control-theoretic translation:** LQR-based minimum-energy rescue trajectories with biophysically-grounded input mapping via TES1.

## Results (N=4 subjects Miller ECoG · N=9 subjects Boran iEEG)

### Miller N-back ECoG (primary cohort)

| Analysis | Key result | Interpretation |
|----------|-----------|----------------|
| **Preprocessing** | ch rejected: al 3/40, cc 3/64, ug 12/62, ca 1/64 | MAD-based; ug has highest artefact rate |
| **PR per subject** | cc: 4.80→5.18→5.44; al: 5.43→5.64→5.43; ca: 4.00→3.87→3.81; ug: 3.80→3.58→3.67 | Two subjects (cc, al) show positive trend; ca and ug show decrease |
| **LME PR~load** | β=0.046, p=0.352, R²<0.001 (N=4 subjects, permutation) | **Null result**: load-dependent PR expansion does not survive mixed-effects test |
| **8-PC variance** | al: 38.9%, ca: 60.3%, cc: 44.2%, ug: 69.5% | Wide subject variability in manifold compactness |
| **LOSO AUROC** | 0.603 (p=0.024, permutation) | Geometry features predict target detection above chance (modest effect) |
| **Tangling Q(t)** | Non-target mean Q > target (7108 vs 5180); 0 significant clusters (FWER) | Mean effect present but not significant after cluster correction; target trials have lower tangling (more stable dynamics) |
| **DMD \|λ\|** | Target: 0.9998, Non-target: 0.9999 | Both near unit circle; negligible condition difference |
| **LQR closed-loop** | E=57.1, final error=1.74 in 150 steps | Feasibility demonstrated; 150-step convergence with 2-channel random B |
| **Cross-subject RSA** | Mean pairwise r=−0.25; cc/ug cluster (r≈1.0); al/ca anti-correlated | Two representational strategies; noise ceiling 0.13–0.25 |
| **CTG max diagonal AUC** | al: 0.767, ca: 0.622, cc: 0.767, ug: 0.773 | Stable WM representations: 0/2-back decodable well above chance at each timepoint |
| **CTG off-diagonal AUC** | cc: 0.639, ug: 0.657 vs al: 0.542, ca: 0.523 | cc/ug have more temporally generalised (activity-maintained) representations |
| **Cross-subject transfer** | LOSO AUC: al 0.539, ca 0.425, cc 0.496, ug 0.502 | Near-chance cross-subject transfer; idiosyncratic geometries |
| **Ring attractor phase** | Phase range ≈2π for all subjects | 2-back maintenance trajectories span full angular range; consistent with ring attractor |

### Boran Sternberg iEEG (replication cohort, N=9)

| Analysis | Key result | Interpretation |
|----------|-----------|----------------|
| **PR vs set size** | Group mean: 4→2.97, 6→2.47, 8→2.69 (non-monotonic); 4/9 subjects show increase | MTL PR does not consistently scale with load; MTL ≠ PFC in load-PR relationship |
| **Error-trial tangling** | sub-02 (N_err=6): Q_err=13,540 vs Q_corr=4,157 | Error trials have 3× higher tangling at probe onset — consistent with H2b |
| **Cross-dataset RSA** | Miller vs Boran group RDM: r=0.60, p=0.112 (Mantel); partial r=0.52 | Trending but not significant; task-general geometric structure suggested |
| **TES1 B matrices** | N=17 subjects; strongest DLPFC coupling: P014A (14.7 mV/mA), weakest: P04 (−1.2 mV/mA) | Patient-specific 10× variation in stimulation coupling — personalised LQR is necessary |

**Key scientific interpretation:** The PR-load null (both Miller and Boran) is the most robust finding and should be reported prominently. The CTG off-diagonal pattern (cc/ug temporally stable vs al/ca time-varying) maps onto the RSA cluster structure and suggests two WM maintenance strategies. The error-trial tangling in Boran is consistent with H2b (errors have higher tangling) but based on N_err=6 in one subject — not definitive. Cross-dataset RSA (r=0.60, p=0.11) provides a suggestive but underpowered test of task-general WM geometry. The TES1 data reveals 10× inter-subject variation in DLPFC coupling, making personalised LQR a practical necessity.

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
│   ├── 01_preprocessing.ipynb       ← HGP, epoching, QC  [EXECUTED → results/01_epochs_*.npz]
│   ├── 02_latent_geometry.ipynb     ← PCA, PR, principal angles  [EXECUTED → results/02_geometry_*.npz]
│   ├── 03_tangling_dynamics.ipynb   ← Q(t) tangling, DMD  [EXECUTED → results/03_dynamics.npz]
│   ├── 04_predictive_model.ipynb    ← LOSO AUROC, electrode capacity  [EXECUTED → results/04_*.npz]
│   ├── 05_control_theory.ipynb      ← LQR rescue, TES1 B matrix  [EXECUTED → results/05_*.npz]
│   ├── 06_paper_figures.ipynb       ← Figure assembly  [EXECUTED → figures/]
│   ├── 07_rsa_neuroai.ipynb         ← Cross-subject RSA, CKA, Procrustes  [EXECUTED → results/07_rsa_results.json]
│   ├── 08_extended_analysis.ipynb   ← Ring attractor, CTG, LOSO transfer  [EXECUTED → results/08_extended_results.json]
│   └── 09_cross_dataset_replication.ipynb  ← Boran replication (N=9), TES1, error-trial geometry  [EXECUTED → results/09_*.npz]
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
