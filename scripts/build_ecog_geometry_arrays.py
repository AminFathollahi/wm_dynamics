#!/usr/bin/env python3
"""Principal-component trajectories and maintenance-window geometry of the ECoG n-back corpus.

Reads results/01_epochs_{subject}.npz and writes results/02_geometry_{subject}.npz
(pr_per_trial, var_ratio, Z, task_id, tgt_id, times, theta_tgt_vs_ntgt) plus a
code_identity record. Read by the cross-temporal, divergence, closed-loop and
figure producers.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from geometry import nback_geometry_arrays  # noqa: E402
from preprocessing import SUBJECTS  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

RESULTS = ROOT / "results"
N_COMPONENTS = 8
N_ANGLE_DIMS = 4
MAINT_WINDOW = (0.3, 1.4)


def main():
    identity = canonical_json(code_identity(ROOT, Path(__file__)))
    for subject in SUBJECTS:
        epochs = np.load(RESULTS / f"01_epochs_{subject}.npz")
        arrays = nback_geometry_arrays(
            epochs["epochs"], epochs["times"], epochs["task_id"], epochs["tgt_id"],
            maint_window=MAINT_WINDOW, n_components=N_COMPONENTS, n_angle_dims=N_ANGLE_DIMS,
        )
        out = RESULTS / f"02_geometry_{subject}.npz"
        np.savez(out, **arrays, code_identity=np.array(identity))
        print(f"{subject}: Z {arrays['Z'].shape} -> {out}")


if __name__ == "__main__":
    main()
