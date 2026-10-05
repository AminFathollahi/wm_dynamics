# wm_dynamics — Population dynamics of working-memory maintenance

A neural data-science codebase for asking how working memory is held in
population activity during the delay period, and what that implies for
stimulation. The organising question is practical: to improve working memory
with closed-loop stimulation driven by a decoder of the neural state, where
should one record, where should one stimulate, when, and at what intensity —
and what do recordings made before stimulation say about those choices?

The analyses run across human, macaque and mouse recordings, from single units
to depth field potentials, ECoG and scalp EEG. Every conclusion is scoped to
its recording method, task, region and time window. Results are reproducible
end to end: each result file records the code, parameters, data identity and
random seeds that produced it, and figures are rebuilt from those files.

## Status (5 October 2026)

The data, preprocessing and analysis code are in place for all datasets
listed below, and the analysis plan has been fixed in advance: for each
hypothesis and dataset it names one primary test, the neurons used, the
state space and the smallest effect the data could detect. Of 294 hypothesis-by-dataset combinations, 99 can be
tested with the available data; the rest are recorded with the reason they
cannot. We are now running those tests, starting with the human recordings.

### Reproducing and extending a hippocampal-stimulation study

A 2025 preprint on hippocampal stimulation during working memory
(bioRxiv, doi:10.1101/2025.08.20.671301) decoded the remembered picture
category from category-selective neurons, described the population with
demixed principal component analysis (dPCA), and measured how far each
trial's activity sat from its category's resting point. Its stimulation
recordings are not public, so we applied the same methods to the public
human recordings without stimulation, and added nested cross-validation,
patient-level statistics and a side-by-side comparison of all neurons
versus selected neurons.

The best read-out, chosen by nested cross-validation among 36 options, is a
linear classifier on each neuron's spike count averaged over the delay, with
category-selective neurons pooled across patients. Neurons are selected on
training trials only. With five categories, chance accuracy is 0.20 and
chance AUC is 0.50.

| Dataset | Neurons | Balanced accuracy | AUC |
|---|---|---|---|
| DANDI 000469 | about 48 | 0.91 | 0.98 |
| DANDI 001187 and 000673 | about 88 | 0.39 | 0.71 |
| DANDI 000004 (recognition task, 0.5 s blank after the picture) | selective | 0.11 above chance | 0.09 above chance |

Pooling neurons from many patients removes the noise that neurons recorded
together share, so these figures are a best case. Within a single recording
session the same classifier is at most 0.03 above chance.

Decoding over time tells a consistent story. In DANDI 000469 the category is
clearest while the picture is on screen (0.74 above chance), still readable
early in the delay (0.32) and weak by its end (0.10). A classifier trained on
the picture period keeps working in the delay (0.28 above chance), so the
delay activity reuses the code formed at encoding. Amygdala neurons carry
more of it than hippocampal neurons (0.23 versus 0.04 above chance). In the
second dataset group the delay signal is weak (about 0.06), and with three
pictures in memory the first one is close to chance. In scalp EEG (two
experiments, 30 and 18 participants) the remembered orientation can also be
decoded, but only about one percentage point above chance, for example
+0.012 (95% interval 0.008 to 0.016, p = 0.0001) before the visual impulse.

We also asked whether common latent-variable models keep this information.
Each model was fitted to one session at a time and compared, trial draw by
trial draw, with the raw spike counts it was built from. Linear models
(principal components, factor analysis, Gaussian-process factor analysis,
dPCA) keep about as much as the raw counts while using a third of the
dimensions or fewer; every difference is within 0.15 and every interval
includes zero. Non-linear and dynamical models keep less. The clearest case
is the larger dataset group (38 patients): diffusion embedding loses 0.20
(interval 0.04 to 0.27, p = 0.03) and a switching linear dynamical system
0.15 (0.02 to 0.26, p = 0.02). None of the latent spaces matches the
selected neurons themselves.

While checking our dPCA against the method the preprint used, we found two
differences: our version did not remove the time course shared by all
categories before fitting, and it read components out with the wrong set of
weights. We deleted every dPCA result in the project and refitted them with
the standard method, verifying that no other number changed.

### Comparing latent representations

A benchmark compares ten ways of summarising population activity (see
below) on held-out neurons and held-out trials. In human single units,
Gaussian-process factor analysis does better than both principal components
and dPCA in the amygdala and with both regions combined, and factor analysis
does better than dPCA in the amygdala. These models are now being tested in
the setting a real-time decoder would face, where only past data are
available at each moment.

### Stability of the working-memory code in ECoG

In an ECoG n-back task (0-back versus 2-back), a decoder trained at one
moment of the delay still works at other moments in all four participants.
The gain in AUC away from the training time ranges from 0.08 to 0.16, with
permutation p-values from 0.003 to 0.0001 (10,000 permutations, each
repeating the full cross-validated analysis).

### Next

- Whether the category axes found during encoding stay in place through the
  delay, in all human single-unit datasets.
- The preprint's distance-to-attractor analysis, reproduced with the
  corrected dPCA.
- The stimulation analyses: how stimulation changes firing, population state
  and behaviour, and how those changes depend on site, timing, dose and the
  brain state before stimulation.

## Questions the code addresses

- **Maintenance state.** Is there a trial-specific, cross-neuron population
  state during the delay? Tested against permutation nulls that keep each
  neuron's own temporal statistics and break only the alignment between
  neurons.
- **Content and selectivity.** What the delay activity encodes (item
  category, identity, load, previous trials); category- and identity-selective
  neurons, their firing rates, and how population decoding depends on them.
  Each analysis states whether it uses all recorded neurons or a selected
  subset of selective neurons.
- **Temporal structure.** Autocorrelation, timescales, drift, switching,
  rotation and cross-temporal generalisation of delay activity.
- **Geometry.** Subspaces, principal angles between conditions, sessions and
  regions, dimensionality, and demixed components (condition-, time- and
  interaction-specific axes).
- **Behaviour.** How single-neuron firing, decoding accuracy and population
  dynamics relate to accuracy and reaction time, within and between sessions,
  with firing-rate and session-bias controls.
- **Observability.** Where the state can be read out: depth versus scalp,
  frequency bands, spikes versus field potentials, and human versus animal
  preparations.
- **Stimulation.** How direct electrical, transcranial and microstimulation
  change firing rates, decoding and population trajectories, and how they
  recover; effects of site, dose, timing and pre-stimulation state; and
  in-silico closed-loop control on fitted models.

## Latent-representation benchmark

Many questions above depend on a low-dimensional description of the
population. The benchmark compares candidate representations on held-out data
with identical folds and decoders:

- **Candidates:** the native (full-rank) population, principal component
  analysis, factor analysis, demixed principal component analysis, Gaussian
  process factor analysis, a sequential variational autoencoder (lfads-torch),
  a recurrent switching linear dynamical system (lindermanlab/ssm), a neural
  data transformer, a time-contrastive embedding and a temporal diffusion
  embedding. Each non-linear model runs through its original package.
- **Score 1, co-smoothing:** a quarter of the neurons are held out; the model
  predicts their activity from the others. The score is the gain in
  log-likelihood over a constant-rate model, in bits per spike (for field
  potentials, bits per observation under a Gaussian model of band power).
- **Score 2, decodable content:** a linear decoder trained on the latent state
  at one time and tested at the same time on held-out trials; the score is the
  area under the ROC curve minus the same quantity with shuffled labels.
- **Comparison rule:** for each score, candidate minus reference is computed
  per entry, averaged within patient, and given a patient-resampled 95%
  confidence interval. Principal components and demixed principal components
  are both used as references. A candidate is retained if, at every memory
  load, its interval lies above zero on at least one score and its upper end
  reaches zero on the other.
- **Read-out timing:** each model is scored both with whole-trial inference and
  with causal inference (the state at time t uses data up to t only), which is
  what a real-time closed-loop decoder would see.

The same benchmark runs on field potentials (log band power in five bands,
100 ms bins) so that the answers carry over to recordings without single
units.

## Datasets

All data are public and live outside the repository.

| Preparation | Dataset | Signal and task |
|---|---|---|
| Human | DANDI 000469, 000673, 001187 | Single units and field potentials, medial temporal lobe and frontal cortex; picture working-memory task with memory loads 1 and 3 |
| Human | DANDI 000574 | Depth field potentials, scalp EEG and single units; letter Sternberg task, set sizes 4, 6 and 8 |
| Human | OpenNeuro ds004752 | Overlapping release of the DANDI 000574 recordings (BIDS; depth, beamformed cortical sources, scalp EEG) |
| Human | DANDI 000004 | Single units; new/old recognition memory; analysed as its own group, never pooled with the working-memory datasets |
| Human | ECoG n-back corpus | Electrocorticography, 0/1/2-back |
| Human | OpenNeuro ds005489, ds005557 | Intracranial recordings with open-loop and closed-loop stimulation during memory encoding |
| Human | Phase-locked intracranial stimulation corpus (doi:10.1016/j.celrep.2019.10.072) | Depth field potentials with phase-locked stimulation |
| Human | OpenNeuro ds005034 | Scalp EEG, verbal working memory, theta transcranial alternating-current stimulation versus sham |
| Human | OpenNeuro ds006848 | Scalp EEG, digit-span serial recall, 6 s retention |
| Human | Closed-loop transcranial alternating-current stimulation corpus (doi:10.1016/j.brs.2024.07.007) | Scalp EEG with closed-loop stimulation |
| Human | Impulse-perturbation scalp-EEG corpus | Scalp EEG, visual working memory with task-irrelevant impulses |
| Macaque | Prefrontal spatial working-memory corpus (Dryad doi:10.5061/dryad.kkwh70sct) | Lateral prefrontal populations |
| Macaque | Multi-object working-memory corpus (doi:10.64898/2026.01.27.702062) | Frontal single and multi-units, continuous report |
| Macaque | Prefrontal microstimulation corpus (doi:10.1523/JNEUROSCI.1742-24.2025) | Single units with delay-period microstimulation |
| Macaque | CRCNS pfc-3, pfc-4 | Prefrontal single units; delayed match and vibrotactile frequency discrimination |
| Mouse | Anterior lateral motor cortex corpus (doi:10.25378/janelia.7489253) | Single units, delayed response |

The registry is `config/datasets.json`; paths below the data root follow it.

Machine-local paths go in `config/project.local.json`, which is not versioned.
Copy `config/project.local.example.json` to that name and fill in the directory
that holds the dataset subdirectories and, optionally, the interpreters of the
model environments below. Environment variables (`WM_DYNAMICS_DATA_ROOT` and
the others declared in `config/project.json`) override the file.

## Repository structure

```
wm_dynamics/
├── src/               analysis library
│   ├── preprocessing.py      filtering, referencing, epoching, band power
│   ├── spike_pipeline.py     shared single-unit pipeline
│   ├── corpus_sessions.py    one iteration over (dataset, region, session)
│   ├── geometry.py           PCA, dimensionality, subspace angles, RSA, cross-temporal generalisation
│   ├── dynamics.py, drift_dynamics.py, state_persistence.py
│   ├── info_decoding.py      matched-fold decoding
│   ├── stimulation_events.py stimulation timing and parameters
│   ├── causal.py             treatment-effect estimators for stimulation data
│   ├── closed_loop.py, control.py   closed-loop simulation, controllability, LQR
│   ├── statistics.py         bootstrap, permutation and cluster tests, mixed models, meta-analysis
│   └── visualization.py      figure utilities
├── scripts/           one runner per analysis (run_*.py), model workers (fit_*_worker.py),
│                      cross-dataset aggregators and figure builders
├── tests/             tests of scientific and software contracts
├── notebooks/         exploratory notebooks
└── config/            dataset registry and machine-local configuration
```

`src/` is the library. Each script in `scripts/` reads data, calls `src/`, and
writes one result file into `results/` (not versioned). Figures are rebuilt
from `results/`.

## Setup

```bash
conda env create -f environment.yml
conda activate wm_dynamics
cp config/project.local.example.json config/project.local.json   # then edit the paths

python -m pytest tests/ -q
python scripts/run_000469_pipeline.py
python scripts/run_human_cosmoothing_benchmark.py
```

Three benchmark candidates need their own environments, because their
packages pin incompatible dependencies: lfads-torch (`lfads_python`),
lindermanlab/ssm (`ssm_python`) and neural data transformers (`ndt_python`).
Set each interpreter in `config/project.local.json`. They run as subprocesses;
if one is missing, that candidate is recorded as not fitted, with the reason,
and the rest of the benchmark runs.

Long analyses write atomic, resumable checkpoints under
`results/.checkpoints/`. A checkpoint is keyed on the code that produced it, so
editing the code restarts the analysis; run long benchmarks from a frozen copy
of the repository.

## Methods in brief

- **Signal processing.** Bipolar or common-average referencing, line-noise
  notch, band power and high-gamma envelope, epoching on task events, robust
  channel rejection.
- **Cross-validation.** Every learned step (normalisation, neuron selection,
  dimensionality reduction, decoders) is fit on training folds only.
  Regularisation penalties, including the demixing penalty, are chosen by
  cross-validation within the training data (nested cross-validation). Folds
  and seeds are explicit.
- **Statistics.** Inference is at the level of the patient or animal: patient
  or session resampling for confidence intervals, permutation and
  cluster-based permutation tests, linear mixed models (including analysis of
  covariance for stimulation factorials), random-effects meta-analysis across
  datasets, and Benjamini–Hochberg false discovery rate. Results are reported
  as estimates, intervals, sample sizes and p-values.
- **Stimulation.** Effects are estimated within each dataset and never pooled
  across datasets or recording modalities; human data are primary and animal
  data supplementary.

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
7. Panichello MF et al. (2024) Intermittent rate coding and cue-specific
   ensembles support working memory. *Nature*, doi:10.1038/s41586-024-08139-9.
8. Murray JD et al. (2017) Stable population coding for working memory
   coexists with heterogeneous neural dynamics in prefrontal cortex.
   *PNAS* 114:394–399.
