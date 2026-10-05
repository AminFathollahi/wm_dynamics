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

**Where the project stands.** The work runs in fixed stages: data lock,
code, representation benchmark, analysis freeze, re-running stale results,
the causal chain of analyses (stimulation → firing → population state →
behaviour), synthesis, independent audit, and the written report. The first
four stages are complete. The analysis plan is frozen in
`preregistration/analysis_freeze.md`: every hypothesis × dataset cell is
either frozen with one primary test, unit set, state space and power bound
(99 cells) or marked not askable with the reason (195 cells). Stale results
are being regenerated in dependency order (12 of 24 steps remain), after
which the chain analyses start, human data first.

**Reproducing a published stimulation study and extending it.** A 2025
preprint on hippocampal stimulation during working memory (bioRxiv
doi:10.1101/2025.08.20.671301) used category-selective neurons, population
decoding of the remembered picture category, demixed principal component
analysis (dPCA) and a distance-to-attractor measure. Its stimulation data are
not public, so we reproduce every method on the public, stimulation-free human
recordings and extend it with nested cross-validation, explicit unit sets
(all neurons versus selected neurons) and patient-level inference.

- **Decoding the remembered category (load 1, delay period).** The decoder was
  chosen by nested cross-validation over 36 read-outs: a linear support vector
  machine on spike counts averaged over the delay, applied to a
  pseudo-population of category-selective neurons pooled across patients
  (neurons selected on training trials only). Chance balanced accuracy is
  0.20 (5 categories); chance AUC is 0.50.

  | Dataset | Neurons | Balanced accuracy | AUC (label-shuffle null) |
  |---|---|---|---|
  | DANDI 000469 | ~48 selective | 0.908 | 0.976 (0.505) |
  | DANDI 001187 + 000673 | ~88 selective | 0.389 | 0.708 (0.501) |
  | DANDI 000004 (recognition task, 0.5 s post-picture blank) | selective | +0.110 above null [0.068, 0.142] | +0.089 above null |

  These pseudo-population numbers are an upper bound: within a single
  recording session the same decoder is at most 0.03 above chance.
- **Time course.** Content peaks during encoding (000469: 0.74 above chance),
  stays readable in the early delay (0.32), and fades by the end (0.10). A
  decoder trained on encoding still reads the delay (0.28 above chance), so
  the delay reuses the encoding code. In 000469 the amygdala carries it (0.23
  above chance on average) more than the hippocampus (0.04). In
  001187 + 000673 the delay content is weak (about 0.06). With three items in
  memory, the first item is near chance.
- **Scalp EEG** (impulse-perturbation corpus, 30 + 18 participants): the
  remembered orientation is decodable but small, about 1 percentage point
  above chance in each phase (for example +0.012 [0.008, 0.016], p = 0.0001
  before the impulse in experiment 1).
- **Do latent models keep the content?** Each latent model was fitted per
  session, its dimensions pooled across sessions, and compared draw by draw
  with the raw spike counts of the same sessions and trials (patient
  bootstrap). Linear models (principal components, factor analysis,
  Gaussian-process factor analysis, dPCA) keep about what the raw counts
  carry with a third of the dimensions or fewer (differences within ±0.15, all
  intervals include 0). Non-linear and dynamical models keep less, most
  clearly in the larger dataset group (38 patients): diffusion embedding
  −0.195 [−0.27, −0.04], p = 0.03; switching linear dynamical system −0.148
  [−0.26, −0.02], p = 0.02. No latent space reaches the selected neurons
  themselves.
- **dPCA corrected to the published method.** Our earlier dPCA fitted its
  axes to the category averages without first removing the time course shared
  by all categories, and read components out with the encoder rather than the
  decoder. Both differ from the standard method the preprint used. All dPCA
  results in the project have been deleted and refitted with the standard
  method (category marginalisation, decoder read-out, regularisation chosen by
  held-out reconstruction of the category marginal). Each refitted file was
  checked to leave every non-dPCA number byte-identical. With dPCA as the
  reference, 9 more benchmark comparisons now pass the pre-declared rule (26
  instead of 17); no comparison against principal components changed.

**Representation benchmark outcome (human single units).** Gaussian-process
factor analysis passes the co-primary rule against both references in the
amygdala and with regions pooled (against principal components also in the
hippocampus), and factor analysis now passes against dPCA in the amygdala; passers get causal
and chronological read-out follow-ups.

**Cross-temporal stability in ECoG (0- versus 2-back).** A decoder trained at
one time in the delay generalises to other times in all four participants:
mean off-diagonal AUC above chance 0.079–0.164, label-permutation p from
0.0034 to 0.0001 (10,000 permutations, the whole nested analysis rebuilt for
every permutation), all four below a 0.05 false-discovery threshold.

**Running and next.** The remaining stale-result reruns; the
memorandum-axis analysis (does a demixed category subspace found during
encoding persist through the delay); latent and axis analyses on DANDI
000004; the factor-analysis read-out follow-up; the reproduction of the
preprint's distance-to-attractor analysis with the corrected dPCA; then the
stimulation chain analyses and their audit.

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
├── preregistration/   decision rules fixed before the corresponding analyses were run
├── notebooks/         exploratory notebooks
├── provenance/        data lock and dataset-identity records
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
