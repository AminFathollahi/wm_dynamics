#!/usr/bin/env python3
"""Panichello & Buschman 2021 (Nature) — NHP PFC/FEF/parietal color working-memory
dataset. Loader stub; not yet available in this project's data holdings.

This is the highest-value pending dataset for the content/context dissociation:
a continuous-report color working-memory task with simultaneous PFC/FEF/parietal
recordings would give content decoding, axis-rotation/ring geometry, and a
drift-predicts-error test in a single recording system — the system in which the
ring-attractor account of working memory is best supported in the literature —
rather than assembled across the six datasets currently in hand. It is typically
released on request rather than through an open Dryad/Zenodo deposit, so this
module is a loader stub, not a runnable analysis.

Expected path: data/panichello2021/ (mirrored under the same external-drive
data root as the other datasets; see DATA_DIR below).

Outstanding implementation work once the data is available:
  - Loader: adapt load_spike_times/build_psth from src/spike_pipeline.py to this
    dataset's spike-time format. The release schema should be confirmed before
    assuming NWB; Panichello & Buschman's public releases are typically MATLAB
    trial structs.
  - Content decoding: decode the remembered color (continuous-report, likely
    discretized into classes) within a fixed set size across the delay, using
    item_identity_ctg / ctg_content_permutation_null (src/geometry.py).
  - Axis-rotation analysis: coding_direction_stability / time_resolved_stability
    (src/geometry.py) on the color-coding axis, compared against the
    context/set-size axis stability computed for the other datasets.
  - Contraction-behavior link: ensemble_dmd (src/dynamics.py) contraction rate
    correlated with recall-error magnitude. This dataset's primary behavioral
    variable is continuous report error rather than a binary correct/error
    outcome, which is a richer measure than what is available elsewhere in
    this project.
  - The numerical pipeline should be reused directly from src/spike_pipeline.py,
    src/geometry.py, and src/dynamics.py; only the file-format adapter in this
    script should be dataset-specific, consistent with every other dataset
    driver script in this project.

Run:
    conda run -n wm_dynamics python scripts/run_panichello_pipeline.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = Path(
    "/media/amin/EXTERNAL_USB/SMAF/Research/Representation/Working Memory/data/panichello2021"
)


def main():
    if not DATA_DIR.exists():
        print(f"SKIP - Panichello & Buschman 2021 data not found at {DATA_DIR}.")
        print("This dataset is typically available on request rather than through an "
              "open deposit. Implement the loader (see module docstring) once it has "
              "been obtained.")
        return
    raise NotImplementedError(
        "Panichello & Buschman 2021 data is present but the loader is not yet "
        "implemented; see the module docstring for the required pipeline."
    )


if __name__ == "__main__":
    main()
