#!/usr/bin/env python3
"""Pooled trajectory tangling and dynamic-mode eigenvalues of ECoG n-back 2-back trials.

Reads results/02_geometry_{subject}.npz and writes results/03_dynamics.npz
(Q_tgt_pool, Q_ntgt_pool, evals_tgt, evals_ntgt) plus a code_identity record. Read by the
tangling cluster test and the figure producer.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamics import pooled_nback_dynamics  # noqa: E402
from preprocessing import SRATE, SUBJECTS  # noqa: E402
from provenance import canonical_json, code_identity  # noqa: E402

RESULTS = ROOT / "results"
DT = 1.0 / SRATE
DMD_RANK = 6
MAINT_WINDOW = (0.3, 1.4)


def main():
    geometry = {}
    for subject in SUBJECTS:
        file = np.load(RESULTS / f"02_geometry_{subject}.npz", allow_pickle=True)
        geometry[subject] = {key: file[key] for key in ("Z", "task_id", "tgt_id", "times")}
    arrays = pooled_nback_dynamics(geometry, DT, MAINT_WINDOW, DMD_RANK)
    if not arrays:
        raise SystemExit("no condition had enough trials; nothing written")
    out = RESULTS / "03_dynamics.npz"
    identity = canonical_json(code_identity(ROOT, Path(__file__)))
    np.savez(out, **arrays, code_identity=np.array(identity))
    print(f"{sorted(arrays)} -> {out}")


if __name__ == "__main__":
    main()
