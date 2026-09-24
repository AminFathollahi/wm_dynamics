# wm_dynamics — Population dynamics of working-memory maintenance

A neural data-science toolkit for asking how working memory is maintained in
population activity during the delay period: is there a trial-resolved,
cross-neuron state, what does it encode, how does it evolve over time, does it
predict behavior, and how does it respond to stimulation?

The code works across human, macaque, and mouse recordings, keeping every
conclusion scoped to its recording method, task, and time window. Analyses are
reproducible end to end: each result file records the script and parameters
that produced it, and figures are rebuilt from those result files.

## Scientific scope

- **Maintenance state.** Tests for a trial-specific, cross-unit population
  state during the delay, against permutation nulls that preserve each
  neuron's own temporal statistics but shuffle units against each other.
- **Temporal structure.** Autocorrelation, timescales, drift, switching,
  rotation, and cross-temporal generalization of delay activity.
- **Content and geometry.** What the dominant delay subspace encodes
  (memorandum, context, previous trials), subspace alignment and angles
  between conditions, sessions, and brain areas, and dimensionality
  (participation ratio, rank estimates).
- **Behavior.** Links between delay geometry or persistence and trial accuracy
  or reaction time, with firing-rate controls.
- **Observability.** Where the state is measurable: depth vs. scalp
  recordings, frequency bands, spikes vs. field potentials, and human vs.
  animal preparations.
- **Stimulation and control.** Response to direct electrical stimulation
  (site, dose, timing, pre-stimulation state), controllability analysis, and
  in-silico closed-loop simulations on fitted models.

## Datasets

Primary single-unit and population cohorts:

| Preparation | Dataset | Task / signal |
|---|---|---|
| Human | Rutishauser lab Sternberg (DANDI 000469, 001187, 000673) | Single units, Sternberg working memory |
| Human | Boran Sternberg (DANDI 000574) | Intracranial EEG + single units, set sizes 4/6/8 |
| Human | Miller N-back | ECoG, prefrontal/parietal, 0/1/2-back |
| Macaque | Panichello et al. 2024 (Dryad doi:10.5061/dryad.kkwh70sct) | Lateral prefrontal populations, working memory / attention |
| Macaque | CRCNS pfc-3 | Prefrontal units, delayed match |
| Mouse | Inagaki et al. 2019 | Anterior lateral motor cortex, delayed response |

Stimulation and methods-support data:

| Dataset | Signal | Use |
|---|---|---|
| Macaque prefrontal microstimulation | Delay-period microstimulation | Causal perturbation response |
| RAM (ds005489) | Human intracranial EEG, stimulation | Encoding-period stimulation response |
| TES1 (Huang et al. 2017) | Transcranial stimulation fields | Forward-model support |

All datasets live outside the repository. Machine-local paths are resolved
through `config/project.json` and `config/datasets.json`. Point
`WM_DYNAMICS_DATA_ROOT` at the directory containing the dataset
subdirectories.

## Repository structure

```
wm_dynamics/
├── src/                 ← Analysis library
│   ├── preprocessing.py ← Filtering, referencing, epoching, spectral power
│   ├── spike_pipeline.py← Shared single-unit Sternberg pipeline
│   ├── geometry.py      ← PCA, dimensionality, subspace angles, RSA, cross-temporal generalization
│   ├── dynamics.py      ← State-space models, DMD/Koopman, drift, tangling
│   ├── state_persistence.py, drift_dynamics.py, subspace_identity.py
│   ├── info_decoding.py ← Matched decoder benchmark (native, PCA/FA, GPFA, CEBRA, LFADS, ...)
│   ├── causal.py        ← Treatment-effect estimators for stimulation data
│   ├── closed_loop.py / control.py ← In-silico closed-loop simulation, controllability, LQR
│   ├── statistics.py    ← Bootstrap, permutation and cluster tests, mixed effects, meta-analysis
│   ├── observability.py, variance_partition.py, selectivity_test.py, ...
│   └── visualization.py ← Publication figure utilities
├── scripts/             ← Per-dataset pipelines (run_<dataset>_<analysis>.py) and cross-dataset aggregators
├── tests/               ← Regression tests for scientific and software contracts
├── notebooks/           ← Exploratory analyses 01–09
├── config/              ← Dataset registry and machine-local path configuration
├── results/             ← Analysis outputs (.json / .npz, generated locally, not versioned)
└── figures/             ← Generated figures (PDF/PNG, generated locally, not versioned)
```

`src/` is the stable library; `scripts/` contains one runner per
dataset/analysis that reads data, calls `src/`, and writes an artifact into
`results/`. Cross-dataset syntheses live in `scripts/aggregate_*.py`, and
figure scripts rebuild all figures from `results/`.

## Quickstart

```bash
conda env create -f environment.yml
conda activate wm_dynamics

# Point at your local data directory (dataset subfolders keep the
# relative layout registered in config/datasets.json)
export WM_DYNAMICS_DATA_ROOT=/path/to/wm-data

# Check the installation
python -m pytest tests/ -q

# Run one pipeline (each writes its artifact into results/)
python scripts/run_000469_pipeline.py
python scripts/run_panichello_pipeline.py
python scripts/run_state_persistence.py
```

Figure scripts in `scripts/` rebuild all figures from whatever artifacts are
present in `results/`.

Long-running jobs write atomic incremental checkpoints under
`results/.checkpoints` and can be resumed. Learned preprocessing and
representations are fit on training data only, with explicit seeds and
split boundaries; each analysis adds a focused regression test.

## Methods in brief

- **Signal processing.** Common-average referencing, line-noise notch,
  high-gamma power (70–150 Hz envelope), epoching on task events, per-channel
  baseline normalization, robust channel rejection.
- **Population geometry.** PCA on the maintenance window, participation ratio,
  principal angles between subspaces, representational similarity, and
  cross-temporal generalization with nested cross-validation and
  label-permutation nulls.
- **Dynamics.** Linear state-space models with separate process and
  observation noise, lag-excluding moment fits, DMD / Koopman extensions,
  drift and switching adjudication, held-out scoring at the patient or
  session level.
- **Decoding benchmark.** Matched-fold comparison of native, linear
  (PCA/FA/GPFA), and nonlinear (CEBRA, T-PHATE, LFADS, autoencoders)
  representations with identical decoders and folds.
- **Statistics.** Temporal cluster permutation, percentile bootstrap
  confidence intervals, leave-one-session-out generalization, effect sizes
  (Cohen's d / Hedges' g), linear mixed effects, Benjamini-Hochberg FDR.
- **Stimulation.** Cross-fit treatment-effect estimation, latency and
  state-excursion contrasts, site/dose/timing maps, and closed-loop
  simulation with held-out readouts and mismatched design/evaluation plants
  as anti-circularity guards.

## Key references

1. Russo AA et al. (2018) Motor cortex embeds muscle-like commands in an
   untangled population response. *Neuron* 97:953.
2. Libby A & Buschman TJ (2021) Rotational dynamics reduce interference
   between sensory and memory representations. *Nat Neurosci* 24:715.
3. Panichello MF & Buschman TJ (2021) Shared mechanisms underlie the control
   of working memory and attention. *Nature* 592:601.
4. Inagaki HK et al. (2019) Discrete attractor dynamics underlies persistent
   activity in the frontal cortex. *Nature* 566:212.
5. Ezzyat Y et al. (2018) Closed-loop stimulation of temporal cortex rescues
   functional networks and improves memory. *Nat Commun* 9:365.
6. Maris E & Oostenveld R (2007) Nonparametric statistical testing of
   EEG- and MEG-data. *J Neurosci Meth* 164:177.
7. Huang Y et al. (2017) Measurements and models of electric fields during
   transcranial stimulation. *eLife* 6:e18834.
8. Panichello MF et al. (2024) Intermittent rate coding and cue-specific
   ensembles support working memory. *Nature*, doi:10.1038/s41586-024-08139-9.
9. Murray JD et al. (2017) Stable population coding for working memory
   coexists with heterogeneous neural dynamics in prefrontal cortex.
   *PNAS* 114:394–399.
