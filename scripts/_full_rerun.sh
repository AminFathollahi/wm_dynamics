#!/bin/bash
# One-shot, sequential regeneration of every results/ artifact after the
# correctness fixes (SRATE, Rutishauser MAINT_WIN, permutation p-value +1
# correction, TES1 sigma_mm, etc.). Run once; do not parallelize (avoids any
# lost-update race on results/all_statistics.json, and dependency order below
# matters: base pipelines before scripts that read their .npz outputs).
set -uo pipefail
cd "$(dirname "$0")/.."
source /home/amin/miniconda3/etc/profile.d/conda.sh
conda activate wm_dynamics
LOG=/tmp/full_rerun_log
mkdir -p "$LOG"

run_nb() {
  local nb="$1"
  echo "=== NOTEBOOK $nb $(date) ===" | tee -a "$LOG/00_master.log"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    "notebooks/$nb.ipynb" > "$LOG/nb_$nb.log" 2>&1
  echo "  exit=$? $(date)" | tee -a "$LOG/00_master.log"
}

run_py() {
  local script="$1"
  echo "=== SCRIPT $script $(date) ===" | tee -a "$LOG/00_master.log"
  python "scripts/$script" > "$LOG/py_${script%.py}.log" 2>&1
  echo "  exit=$? $(date)" | tee -a "$LOG/00_master.log"
}

# ---- Phase A: notebooks (foundational Miller/Boran preprocessing+geometry) ----
for nb in 01_preprocessing 02_latent_geometry 03_tangling_dynamics 04_predictive_model \
          05_control_theory 07_rsa_neuroai 08_extended_analysis 09_cross_dataset_replication; do
  run_nb "$nb"
done

# ---- Phase B: base per-dataset pipelines (no cross-script dependencies) ----
for s in run_miller_ctg_corrected.py run_boran_pipeline.py run_000574_units_pipeline.py \
         run_000469_pipeline.py run_001187_pipeline.py run_000673_pipeline.py \
         run_pfc3_content_ctg.py run_panichello_pipeline.py run_wolff_impulse_pipeline.py \
         run_pr_positive_control.py run_dpca_analysis.py; do
  run_py "$s"
done

# ---- Phase C: scripts depending on Phase B outputs (TES1 chain, axis rotation) ----
for s in run_tes1_analysis.py run_divergence_analysis.py run_stim_location_analysis.py \
         run_manifold_rescue_analysis.py run_axis_rotation_analysis.py; do
  run_py "$s"
done

# ---- Phase D: 000469-specific supplementary analyses ----
for s in run_contraction_behavior_analysis_000469.py run_cross_decoding_orthogonality_000469.py \
         run_decoder_confidence_timecourse_000469.py run_full_trial_content_decoding_000469.py \
         run_item_identity_load_generalization_000469.py run_multiitem_ctg_000469.py \
         run_multiitem_recall_decoding_000469.py run_context_confidence_timecourse.py; do
  run_py "$s"
done

# ---- Phase E: multiband (independent, heavy) ----
run_py run_multiband_analysis.py

# ---- Phase F: cross-dataset aggregation (must run last) ----
run_py aggregate_dpca_results.py
run_py aggregate_pr_across_datasets.py
run_py run_context_code_cross_dataset_rsa.py
run_py aggregate_forest_syntheses.py

echo "=== FULL RERUN COMPLETE $(date) ===" | tee -a "$LOG/00_master.log"
