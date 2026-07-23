# WM Dynamics — A Controllable Geometry of Working Memory

**Status:** Pipeline executed | 341/341 tests passing | Nine datasets | Manuscript on the R1–R5 arc (`PAPER_REPORT.tex`, tracked in Overleaf)

## Scientific Question

Working memory must hold a memorandum stably enough to guide behaviour seconds later, yet flexibly enough to be updated. Does it do so through a stable, persistent code or a dynamic one — and does the population geometry of maintenance define a **control policy** that (i) predicts how stimulation should act and (ii) can steer the maintained state?

We characterize the geometry and dynamics of maintenance, read a control policy off it, test that policy against real stimulation, and stress-test it in a closed-loop simulation.

## The arc (five findings)

| | Claim | Key result |
|---|---|---|
| **R1** | WM holds a **stable context code** and a **rotating memorandum** | Context CTG generalizes across the delay in all six neural cohorts; within the same neurons the content axis rotates more than the context axis (paired diff 0.102, *p*=0.008) |
| **R2** | The maintenance manifold is **low-dimensional with single-trial-identifiable dynamics** | PR does not expand with load (pooled slope 0.01, 95% CI [−0.10, 0.12], *p*=0.87); DMD eigenvalues near the unit circle; the trial-averaged mean manufactures a spurious contraction that vanishes under single-trial ensemble estimation |
| **R3** | The fitted dynamics **specify a control policy, and delay-period stimulation obeys it** | Doubly-robust, geometry-conditioned CATE on macaque DLPFC microstim: effect scales with alignment to the unstable direction v\* (slope +0.14, 95% CI [0.08, 0.20], permutation *p*=2×10⁻⁴; 11 sessions, 15,670 trials) |
| **R4** | A closed-loop controller **stabilizes the memorandum in silico** | On the 8/13 cohorts where a 20°-mismatched controller stabilizes the true plant, drift is reduced in **8/8**; held-out decodability improves in **4/6** cohorts with an above-chance decoder |
| **R5** | The stabilization is **robust to noise, moderately to model error, fragile to nonlinearity** | Benefit degrades gracefully with observation noise and coupling mismatch but has a defined operating regime; **5/13** cohorts destabilize under the single 20° draw (ρ(A−BK) > ρ(A)) |

**Honesty guardrails baked into the results:** the dynamic-vs-stable stimulation contrast is a *trend*, not significant (0.075 vs 0.056, *p*=0.10); the human encoding-period stimulation cohort (RAM ds005489) is a same-signed scope-bounded **null** (slope +0.01, *p*=0.57); the closed-loop demonstration is **in silico only** and its benefit is real but partial. No claim of enhanced memory — a computational proof-of-concept that the representational state can be steered.

## Architecture — three arms, linked only by geometry

- **Arm 1 — Observational (7 cohorts):** fit the plant (A, v\*, flow, content/context subspaces) and derive the control policy.
- **Arm 2 — Interventional (2 stimulation cohorts):** test the policy causally (stimulation effect scales with alignment-to-v\*, doubly-robust CATE).
- **Arm 3 — In-silico (new):** close the loop on the fitted plant and stress-test it.

The arms link **only** through the geometric feature map (alignment-to-v\*), never by pooling raw data.

## Datasets (nine + methods-support)

Seven observational + two electrical-stimulation cohorts. TES1 supplies per-subject **B** matrices (methods-support, not counted as a cohort).

| Dataset | N | Signal / Task | Role |
|---------|---|---------------|------|
| **Miller N-back** | 4 subj | ECoG 1000 Hz PFC/parietal · 0/1/2-back | observational |
| **Boran Sternberg** (DANDI 000574) | 9 subj | iEEG + single units · set sizes 4/6/8 | observational |
| **DANDI 000469 / 001187 / 000673** | single units | Rutishauser-lab Sternberg WM | observational |
| **CRCNS pfc-3** | units | macaque PFC delayed match | observational |
| **Soldado-Magraner 2025** | 11 sessions | macaque DLPFC **delay-period microstim** | interventional (causal anchor) |
| **RAM ds005489** | human iEEG | **encoding-period stimulation** | interventional (scope-null) |
| **TES1** (Huang et al. 2017, eLife) | 17 subj | tES-induced fields → **B** matrices | methods-support |

Data on external USB: `/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/`.

## Repository Structure

```
wm_dynamics/
├── src/                       ← Production library (git-tracked)
│   ├── preprocessing.py       ← CAR, notch, high-gamma, epoching (Miller + Boran + TES1)
│   ├── geometry.py            ← PCA, participation ratio, principal angles, CTG, RSA, drift
│   ├── dynamics.py            ← Tangling Q(t), exact/ensemble DMD, EDMD/Koopman, SINDy, flow divergence
│   ├── control.py             ← Controllability, LQR/DARE, minimum-energy control
│   ├── closed_loop.py         ← In-silico closed-loop simulation + robustness sweep (R4/R5)
│   ├── causal.py              ← Cross-fit nuisances, AIPW ATE/CATE, DR-Learner, DML, E-value (R3)
│   ├── statistics.py          ← Bootstrap, cluster permutation, AUROC, LOSO, Hedges' g, LME, forest meta
│   ├── spike_pipeline.py      ← Shared single-unit Sternberg WM pipeline (Rutishauser-lab datasets)
│   ├── io_utils.py            ← Locked concurrent read-modify-write for results/all_statistics.json
│   ├── neuroai.py             ← CKA, Procrustes alignment
│   └── visualization.py       ← Nature Neuroscience–style figure utilities
│
├── notebooks/                 ← Exploratory pipeline 01–09 (git-tracked)
├── scripts/                   ← Per-dataset production pipelines + aggregators (37 scripts)
│   ├── run_*_pipeline.py       ← One runner per dataset (ID-specific naming)
│   ├── run_soldado_pipeline.py, run_ram_openloop_pipeline.py  ← causal arm
│   ├── run_closed_loop_analysis.py                            ← R4/R5
│   ├── aggregate_*.py          ← Pool across datasets (forest syntheses, dPCA, PR)
│   └── generate_paper_figures.py                              ← Assemble all paper figures from results/
│
├── tests/                     ← Unit tests (341 passing; one test file per src module)
├── results/                   ← Saved .npz / .json artifacts (gitignored) — the source of truth for every number
├── figures/                   ← Saved PDF/PNG figures (gitignored)
├── learning/                  ← Pedagogical notebooks (gitignored — local only)
├── PAPER_REPORT.tex           ← Manuscript source (gitignored — tracked in Overleaf)
└── environment.yml            ← Conda environment (activate: conda activate wm_dynamics)
```

## Quickstart

```bash
conda activate wm_dynamics
cd "/home/amin/Research/Representation/Working Memory/wm_dynamics"

# Tests
python -m pytest tests/ -q          # 341 passing

# Per-dataset pipelines write to results/ (require external USB data)
python scripts/run_soldado_pipeline.py        # causal anchor (R3)
python scripts/run_closed_loop_analysis.py    # in-silico loop + robustness (R4/R5)

# Rebuild every figure from the current artifacts
python scripts/generate_paper_figures.py
```

`scripts/_full_rerun.sh` runs the per-dataset pipelines and then the aggregators so
`results/forest_syntheses.json` is rebuilt last from fresh inputs.

## Methods Summary

- **Signal processing:** CAR → notch (60/120/180/240 Hz) → high-gamma power (70–150 Hz Butterworth → Hilbert envelope² → 50 ms Gaussian) → epoch → per-electrode baseline z-score; MAD-based channel rejection.
- **Geometry:** PCA (full SVD) on the maintenance window; participation ratio PR = (Σλ)²/Σλ²; principal angles (Björck & Golub 1973); cross-temporal generalization (CTG) with nested CV and a label-permutation null.
- **Dynamics:** trajectory tangling Q(t) (Russo et al. 2018); exact and single-trial-ensemble DMD (Tu et al. 2014); flow divergence Σlog|λ|/dt; **dt is derived per cohort from the times vector** (Miller 1000 Hz, Boran 1398 Hz).
- **Control:** DARE via `scipy.linalg.solve_discrete_are`; LQR gain K; minimum-energy control via controllability Gramian; TES1 anatomical **B** by Gaussian interpolation from MNI coordinates.
- **Causal (R3):** cross-fit AIPW; geometry-conditioned CATE (`cate_vs_modifier_slope`), DR-Learner, DML partial-linear, permutation inference, E-value sensitivity.
- **Closed-loop (R4/R5):** simulate a fitted plant with the loop on vs off from matched noise draws; controller designed on a mismatched (A_hat, B_hat), scored on a decoder trained only on real uncontrolled trials, benefit reported with bootstrap CIs. Anti-circularity guardrails: (1) estimated-plant controller, true-plant evaluation; (2) held-out read-out; (3) no near-ceiling decoding.
- **Statistics:** temporal cluster-permutation (Maris & Oostenveld 2007); percentile bootstrap CIs (B≥5000); LOSO; Cohen's *d* / Hedges' *g*; linear mixed effects; permutation *p*-floor (c+1)/(n+1); Benjamini-Hochberg FDR.

## Key References

1. Russo AA et al. (2018) *Motor cortex embeds muscle-like commands in an untangled population response.* **Neuron** 97:953.
2. Tu JH et al. (2014) *On dynamic mode decomposition.* **J Comput Dyn** 1(2).
3. Libby A & Buschman TJ (2021) *Rotational dynamics reduce interference between sensory and memory representations.* **Nat Neurosci** 24:715.
4. Panichello MF & Buschman TJ (2021) *Shared mechanisms underlie the control of working memory and attention.* **Nature** 592:601.
5. Inagaki HK et al. (2019) *Discrete attractor dynamics underlies persistent activity in the frontal cortex.* **Nature** 566:212.
6. Ezzyat Y et al. (2018) *Closed-loop stimulation of temporal cortex rescues functional networks and improves memory.* **Nat Commun** 9:365.
7. Maris E & Oostenveld R (2007) *Nonparametric statistical testing of EEG- and MEG-data.* **J Neurosci Meth** 164:177.
8. Huang Y et al. (2017) *Measurements and models of electric fields during tES.* **eLife** 6:e18834.
