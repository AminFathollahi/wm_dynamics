# WM Dynamics — A Trial-Specific, Cross-Unit Population State in Human Working-Memory Maintenance

**Status (2026-08-31):** the project is a reproducible evidence base for a proposed model-guided,
closed-loop working-memory stimulation experiment. Its reader-facing scientific record is
[`PAPER_REPORT.tex`](PAPER_REPORT.tex); result JSON files and their producer scripts are the auditable
source of every number. Several historical analyses remain available, but are not current headline
evidence unless the paper identifies them as such.

## Scientific Question

How can a working-memory intervention use neural recordings to choose where to record, where and when to
stimulate, and whether a state measured before stimulation predicts the appropriate site, intensity, or
timing? The project answers parts of that question across human, macaque, and mouse preparations while
keeping each conclusion scoped to its instrument, task, and epoch.

## Start Here

- Read [`PAPER_REPORT.tex`](PAPER_REPORT.tex) for the scientific motivation, methods, all results,
  limitations, and the next falsifiable experiment.
- Read [`docs/BCI_QUESTION_EVIDENCE_MAP.md`](docs/BCI_QUESTION_EVIDENCE_MAP.md) to navigate evidence by
  the four engineering questions.
- Use [`docs/mandates/STATUS.md`](docs/mandates/STATUS.md) for the single operational queue and
  resumable execution state. `REMAINING_WORK.md` is a historical planning snapshot.
- Find a result in `results/`, then use the `producer`/`script` recorded in its metadata or the
  evidence map to locate the reproducible analysis.

## Historical core-result snapshot

The six findings below are retained as a compact account of the project's original core. They are
not a complete statement of the current scientific record; `PAPER_REPORT.tex` controls where later
analyses, robustness checks, or withdrawals changed their interpretation.

Six controlled findings (F1-F6), every existence and shape claim scored against a per-unit
permutation null that reshuffles units against one another while preserving each unit's own temporal
statistics, so a surviving contrast is specifically cross-unit rather than single-unit
autocorrelation.

- **F1 -- a trial-specific, cross-unit population state exists above the counting floor.** DANDI
  000469 (72 session/epoch pairs, 100 ms bins, 3-bin window): the contrast clears FDR contiguously
  from 0.3 s to 2.2 s of the delay (20/25 lags tested; mean 0.0771 at 0.3 s, q=4.2e-04, and 0.0586 at
  2.2 s, q=0.028); the five longest lags tested (2.3-2.7 s) do not clear. Existence is robust at all
  three tested window widths. `results/state_persistence_lag.json`, `results/state_persistence.json`.
- **F2 -- a fast human autocorrelation component, and no resolvable decay established beyond it.**
  Early segment 0.3-0.8 s: mean slope -0.11890, one-sided p=0.0001, n=72. Late segment 0.8-2.7 s:
  mean +0.01727, p=0.9411; a double-difference test between them is itself not significant
  (p=0.4192), so this is a floor bound at this sensitivity, not a plateau -- "a persistent
  component", "a stable state" and "activity-silent's opposite" are not licensed. Mouse ALM does not
  show this fast component at the matched deciding window (p=0.220, n=23); a macaque arm's contrast
  rises rather than declines. A five-rung planted-timescale recovery ladder (0.3-5.0 s) converges on
  every rung without hitting a fit bound, yet recovered-vs-planted rank correlation is rho=-0.30
  (p=0.68, n=5): no tau in seconds is reported. `results/state_persistence_shape.json`.
- **F3 -- whether the memorandum is in the state has opposite, individually significant answers.**
  Leave-one-latent-out deletion-cost observable, chance 0.5, k=8 latents matched: macaque lateral PFC
  (Panichello 2024, 25 sessions across 3 animals in at least 2 prefrontal areas) content is *not* in
  the dominant state (mean fractional rank 0.6743, p=0.0186), reproducing Murray et al. 2017 rather
  than being novel; mouse ALM (23 sessions) content *is* the dominant state (mean 0.1429,
  p=9.999e-05). The pooled estimate averages these opposite-signed effects and is not a result about
  either corpus. Content was not recoverable from the dominant latent in the human sessions that could
  be tested (DANDI 000469; 8/61 clear their own content-decoder null); a second human corpus (DANDI
  001187) was excluded by a label-granularity criterion rather than a property of its recordings, so
  this is currently a limit on what was asked, not an established property of human recordings.
  `results/state_content_link.json`.
- **F4 -- amplitude, not persistence, tracks behaviour, below this project's own pre-declared bar.**
  Rate-free state geometry predicts trial accuracy (r=-0.0974, p=0.0035, n=11 sessions); persistence
  does not (p=0.168); a per-trial-gain account is rejected. The measured \|r\|=0.0974 is below the
  0.14 minimum-effect threshold this project pre-declared, so the association is significant and its
  magnitude does not clear the bar set for it in advance. `results/rate_free_state_geometry_behavior_link.json`,
  `results/state_behavior_link.json`, `results/behavior_amplitude_rate_controls.json`,
  `results/rank1_gain_temporal_profile_closure.json`.
- **F5 -- where the state is measurable, and a sensor/noise-observable dissociation.** Depth
  electrodes restricted to 8-45 Hz carry the state (mean +0.1185, p=3.7e-10); scalp at the identical
  band/patients/pipeline does not (p=0.589). The same artifact's pre-declared factor-analysis
  observation-noise fraction ranks scalp AT OR BELOW depth in the same sessions (band effect
  p=0.0063, sensor effect p=0.0215): the sensor that looks cleaner on the noise-named observable is
  exactly the one on which the state is not measurable, so low observation noise is not sufficient
  for state observability. LFP and single-unit grains are indistinguishable by paired discordance
  (p=0.754) and the LFP grain is ahead on a margin-aware comparison.
  `results/band_versus_sensor_decomposition.json`, `results/observability_census.json`,
  `results/observability_and_power_census.json`, `results/observability_matched_modality_test.json`.
- **F6 -- the instruments available to attribute these limits to observation noise do not support
  that attribution.** Two quantities this project has called observation noise are uncorrelated
  across 32 real cells (r=-0.005, p=0.98). The factor-analysis fraction does track true noise under
  its assumed diagonal covariance, with a smaller dimensionality confound; a second, cross-validated
  nugget-fraction estimator returns near-exactly zero at true noise as high as 0.80 (diagonal) / 0.40
  (correlated), and the human-vs-mouse gap on this construct is estimator-specific, not established
  as a species difference. `results/latent_model_observation_noise_comparison.json`,
  `results/observation_noise_estimator_construct_validity.json`.

See PAPER_REPORT.tex Results (F1-F6) for the full numbers, controls and bounds behind each
finding. The per-finding artifacts under `results/` are the source of truth for every number;
`results/implementation_report.md` is an operational execution log, not an evidence source.

## Historical analyses

The earlier confinement-rate/switching/rotation/intrinsic-timescale spine, plus earlier CTG, DMD,
perturbation-targeting, LQR, and closed-loop results, are retained under the zero-drop rule as an
exploratory archive (`\part{Retained exploratory results archive}` in PAPER_REPORT.tex). Their
current status and hashes are in `provenance/evidence_ledger.json`. They must not be used as current
headline evidence without a replacement artifact. On the retired spine specifically: M2 beat iid M0
by 0.01318 nats/observation but not matched-flexibility baselines (scale-mixture and free-variance
AR(1) controls both crossed zero), a trial's own history outpredicted the preceding same-condition
trial (population-wide, not working-memory-content-specific), rotation and controlled switching were
not established as current results, and the intrinsic-timescale estimator was voided by a lag-0 bug.
See `results/drift_simulation_gate.json`, `results/human_drift_spine_000469.json`,
`results/switching_adjudication.json`, and `results/rotation_estimator_floor.json` for that archived
work.

## Architecture — three arms, linked only by geometry

- **Arm 1 — Observational (7 cohorts):** fit the plant (A, v\*, flow, content/context subspaces) and derive the control policy.
- **Arm 2 — Stimulation (2 cohorts):** audit whether the design identifies a causal targeting
  estimand, then report only licensed contrasts. macaque PFC microstimulation fails this gate because the released
  correct/error files lack a shared trial/block index; RAM is encoding-period and cross-paradigm.
- **Arm 3 — In-silico (new):** close the loop on the fitted plant and stress-test it.

The arms link **only** through the geometric feature map (alignment-to-v\*), never by pooling raw data.

## Datasets (nine + methods-support)

F1-F6 above are computed on DANDI 000469 (human, 18 patients), Panichello et al. 2024 (macaque
lateral PFC, 25 sessions across 3 animals in at least 2 prefrontal areas, dataset DOI
10.5061/dryad.kkwh70sct), and Inagaki et al. 2019 (mouse ALM, 23 sessions). The table below is the
larger nine-cohort roster used by the archived confinement-rate/switching/rotation spine. Seven
observational + two electrical-stimulation cohorts. TES1 supplies per-subject **B** matrices
(methods-support, not counted as a cohort).

| Dataset | N | Signal / Task | Role |
|---------|---|---------------|------|
| **Miller N-back** | 4 subj | ECoG 1000 Hz PFC/parietal · 0/1/2-back | observational |
| **Boran Sternberg** (DANDI 000574) | 9 subj | iEEG + single units · set sizes 4/6/8 | observational |
| **DANDI 000469 / 001187 / 000673** | single units | Rutishauser-lab Sternberg WM | observational |
| **CRCNS pfc-3** | units | macaque PFC delayed match | observational |
| **Macaque PFC microstimulation 2025** | 11 sessions | macaque DLPFC **delay-period microstim** | causal design no-go; descriptive recovery only |
| **RAM ds005489** | human iEEG | **encoding-period stimulation** | interventional (scope-null) |
| **TES1** (Huang et al. 2017, eLife) | 17 subj | tES-induced fields → **B** matrices | methods-support |

Machine-local paths are resolved through `config/project.json`. Set
`WM_DYNAMICS_DATA_ROOT` to the directory containing the dataset subdirectories; their relative
locations remain registered in `config/datasets.json`.

## Repository Structure

```
wm_dynamics/
├── src/                       ← Production library (git-tracked)
│   ├── preprocessing.py       ← CAR, notch, high-gamma, epoching (Miller + Boran + TES1)
│   ├── geometry.py            ← PCA, participation ratio, principal angles, CTG, RSA, drift
│   ├── dynamics.py            ← Tangling, exact/ensemble DMD, EDMD/Koopman, SINDy, log-volume contraction
│   ├── drift_dynamics.py      ← LGSSM and moment estimators for confined single-trial drift
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
├── scripts/                   ← Per-dataset production pipelines, audits, and aggregators
│   ├── run_*_pipeline.py       ← One runner per dataset (ID-specific naming)
│   ├── run_macaque_pfc_microstimulation_pipeline.py, run_ram_openloop_pipeline.py  ← causal arm
│   ├── run_closed_loop_analysis.py                            ← R4/R5
│   ├── aggregate_*.py          ← Pool across datasets (forest syntheses, dPCA, PR)
│   └── generate_paper_figures.py                              ← Assemble all paper figures from results/
│
├── tests/                     ← Scientific and software regression tests
├── results/                   ← Saved .npz / .json artifacts (gitignored) — the source of truth for every number
├── figures/                   ← Saved PDF/PNG figures (gitignored)
├── learning/                  ← Pedagogical notebooks (gitignored — local only)
├── PAPER_REPORT.tex           ← Manuscript source (gitignored — tracked in Overleaf)
└── environment.yml            ← Conda environment (activate: conda activate wm_dynamics)
```

## Quickstart

```bash
conda activate wm_dynamics
cd /path/to/wm_dynamics

# Required for analyses that read staged datasets. Alternatively set
# paths.data_root in config/project.json (keep machine-specific values uncommitted).
export WM_DYNAMICS_DATA_ROOT=/path/to/wm-data

# Tests
python -m pytest tests/ -q

# Per-dataset pipelines write to results/ (require external USB data)
python scripts/run_drift_simulation_gate.py
python scripts/run_human_drift_spine_000469.py
python scripts/run_human_drift_behavior.py
python scripts/run_hierarchical_confinement_000469.py
python scripts/run_geometry_from_drift_parameters_000469.py
python scripts/run_rotation_adjudication.py
python scripts/run_switching_adjudication.py --replicates 200

# Rebuild every figure from the current artifacts
python scripts/generate_paper_figures.py
```

`config/project.json` also defines results, figures, provenance, checkpoint and auxiliary-tool
locations plus shared non-scientific runtime defaults. `WM_DYNAMICS_CONFIG` can point to a complete
machine-local replacement; the per-path environment variables listed in the file take precedence.
Scripts should resolve datasets with `src.project_config.dataset_path()` rather than constructing
absolute paths.

`scripts/_full_rerun.sh` runs the per-dataset pipelines and then the aggregators so
`results/forest_syntheses.json` is rebuilt last from fresh inputs.

## Methods Summary

- **Signal processing:** CAR → notch (60/120/180/240 Hz) → high-gamma power (70–150 Hz Butterworth → Hilbert envelope² → 50 ms Gaussian) → epoch → per-electrode baseline z-score; MAD-based channel rejection.
- **Geometry:** PCA (full SVD) on the maintenance window; participation ratio PR = (Σλ)²/Σλ²; principal angles (Björck & Golub 1973); cross-temporal generalization (CTG) with nested CV and a label-permutation null.
- **Primary dynamics:** unsmoothed fold-frozen coordinates; scalar LGSSM with distinct process/observation noise; lag-0-excluding OU moment fit; patient-level held-out scoring.
- **Archival dynamics/control:** DMD retained-subspace log-volume contraction, LQR, and controllability-Gramian demonstrations are preserved but do not establish empirical control.
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
9. Panichello MF, Jonikaitis D, Oh YJ, Zhu S, Trepka EB, Moore T (2024) *Intermittent rate coding and cue-specific ensembles support working memory.* **Nature** (6 November 2024), doi:10.1038/s41586-024-08139-9.
10. Murray JD, Bernacchia A, Roy NA, et al. (2017) *Stable population coding for working memory coexists with heterogeneous neural dynamics in prefrontal cortex.* **PNAS** 114:394-399, doi:10.1073/pnas.1619449114.
