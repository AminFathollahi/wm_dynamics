# Information benchmark

`scripts/run_info_benchmark.py` compares native activity, PCA, FA, GPFA, CEBRA, T-PHATE, and LFADS with matched held-out folds. Each representation is fit on full-resolution outer-training trials and evaluated with linear and nonlinear decoders against shared stratified label permutations. `--time-step` thins only the decoder time grid.

The configured data root supplies the Panichello single-item and Watters multi-object corpora. LFADS runs in the isolated interpreter configured as `executables.lfads_python`.

## Completed scope

The completed artifact contains all 28 requested cells: seven representations by two decoders in one Panichello session and one Watters session/item-count level. Each cell uses five shared stratified folds and 100 matched label permutations. There were no method failures.

`auc` is the observed mean area under the ROC curve. `null_auc` is the mean across the matched permutation distribution, and `auc_above_null` is their difference. `p_value` is the one-sided matched permutation p-value. Same-time scores average the cross-temporal matrix diagonal; cross-time scores average its off-diagonal entries.

In the Panichello session, LFADS with the linear decoder had the largest point estimates: same-time AUC 0.772 versus null 0.503 and cross-time AUC 0.765 versus null 0.503, both $p=0.0099$. The native linear values were 0.721 and 0.659, both $p=0.0099$. T-PHATE was at its null with either decoder. In the tested Watters cell, the native linear representation had the largest cross-time point estimate, AUC 0.631 versus null 0.505, $p=0.0099$; LFADS did not clear its matched null. CEBRA was significant in the Panichello session but did not exceed the native point estimate and did not clear the matched null in the Watters cell.

These are cell-level exploratory comparisons. The permutation tests ask whether each representation carries label information; they do not test differences between representations. Method rankings, session-level generalization, seed sensitivity, and item-count dependence remain open.

## Repeated-seed expansion

The queued expansion uses one output and checkpoint directory per representation seed. The sweep averages repeated seeds within each recording/session/item-count cell, then computes matched representation contrasts separately by decoder and metric. Bootstrap intervals resample recording sessions while retaining item-count cells within session; permutation draws are not treated as independent observations.

After the GPU is available, launch the resumable sweep as a persistent user service:

```bash
systemd-run --user --unit=wm-info-benchmark-seeds --collect \
  --property="WorkingDirectory=$(pwd)" \
  --property="StandardOutput=append:$(pwd)/provenance/run_logs/info_benchmark_seeds.log" \
  --property="StandardError=append:$(pwd)/provenance/run_logs/info_benchmark_seeds.log" \
  /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_info_benchmark_seed_sweep.py \
  --seeds 0 1 2 3 4 --benchmark-args --n-splits 5 --n-perm 100 --time-step 5
```

Use `--aggregate-only` to rebuild the aggregate from completed per-seed outputs without running benchmark jobs. A rerun resumes existing per-seed outputs and checkpoints.

```bash
conda activate wm_dynamics
python scripts/run_info_benchmark.py \
  --corpora single_item multi_object \
  --single-item-sessions-limit 1 \
  --multi-object-sessions-limit 1 \
  --multi-object-levels-limit 1 \
  --n-splits 5 \
  --n-perm 100 \
  --time-step 5
```

Use `--smoke` for a bounded native/PCA pipeline check, not inference. The runner atomically updates `results/info_benchmark.json` and `INFO_BENCHMARK.md` after every completed stage. Fold fits, observed scores, and individual null draws are checkpointed; rerunning the same command resumes them, while changed inputs or implementation sources invalidate them.

Long runs can be submitted as a user service so they continue independently of the launching shell:

```bash
systemd-run --user --unit=wm-info-benchmark --collect \
  --property="WorkingDirectory=$(pwd)" \
  --property="StandardOutput=append:$(pwd)/provenance/run_logs/info_benchmark.log" \
  --property="StandardError=append:$(pwd)/provenance/run_logs/info_benchmark.log" \
  /usr/bin/env -u WM_DYNAMICS_DATA_ROOT -u WM_DYNAMICS_LFADS_PYTHON \
  /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_info_benchmark.py \
  --corpora single_item multi_object \
  --candidates native_full_rank principal_components factor_analysis \
  gaussian_process_factor_analysis time_contrastive_embedding \
  temporal_diffusion_embedding sequential_autoencoder \
  --decoders linear nonlinear \
  --single-item-sessions-limit 1 \
  --multi-object-sessions-limit 1 \
  --multi-object-levels-limit 1 \
  --n-splits 5 --n-perm 100 --time-step 5
```

`systemctl --user status wm-info-benchmark` reports service state and `journalctl --user -u wm-info-benchmark -f` follows its journal. Re-running the same command resumes matching checkpoints; changes to code, data, folds, or runtime identity create distinct cache entries.
