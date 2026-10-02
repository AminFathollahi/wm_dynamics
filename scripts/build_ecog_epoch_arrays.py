#!/usr/bin/env python3
"""Baseline-normalised high-gamma epochs of the ECoG n-back corpus, one file per subject.

Writes results/01_epochs_{subject}.npz (epochs, times, task_id, tgt_id, stim_id,
good_channels) plus a code_identity record. Read by the ECoG n-back geometry, tangling
and multiband producers.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import SUBJECTS, load_subject, nback_epoch_arrays  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

RESULTS = ROOT / "results"
PRE_MS = 200.0
POST_MS = 1500.0
BAD_CHANNEL_THRESHOLD_MAD = 3.0


def main():
    identity = canonical_json(code_identity(ROOT, Path(__file__)))
    for subject in SUBJECTS:
        arrays = nback_epoch_arrays(
            load_subject(subject), PRE_MS, POST_MS, BAD_CHANNEL_THRESHOLD_MAD
        )
        out = RESULTS / f"01_epochs_{subject}.npz"
        np.savez(out, **arrays, code_identity=np.array(identity))
        print(f"{subject}: {arrays['epochs'].shape} -> {out}")


if __name__ == "__main__":
    main()
