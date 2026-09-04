#!/usr/bin/env python3
"""Power bound for the rotation-gate null.

results/vstar_fit_selection_factorial.json found the ROTATING fraction (theta
bootstrap CI excludes 0, primary ensemble+argmax|lambda| cell) below the
pre-specified 20% threshold in every cohort. That is not by itself evidence
that rotation is absent: the leading mode is complex (oscillatory) in the
majority of sessions in several cohorts, so a low ROTATING fraction can
equally mean the true rotation frequency is too small to resolve against
each session's own bootstrap dispersion at its trial count. A null needs a
power bound to be a result rather than a missing analysis -- this script
computes, per session and per cohort, the smallest |theta| (and the
corresponding frequency in Hz) that the session's own bootstrap dispersion
of theta could have resolved with 95% power at alpha=0.05 (two-sided).

Method: each session's theta_ci in vstar_fit_selection_factorial.json is a
bootstrap 2.5/97.5 percentile interval; under the usual normal approximation
for a bootstrap-percentile CI, SE(theta) = (hi - lo) / (2 * 1.96). The
smallest true |theta| detectable with 95% power at alpha=0.05 (two-sided) is
then delta_min = SE(theta) * (z_{0.975} + z_{0.95}) = SE(theta) * 3.605 --
the standard two-sample-free power/detection-boundary formula for a z-test.
dt is read directly from the same per-session iterators
(run_vstar_eigen_audit.ALL_ITERS) used to build the factorial in the first
place, not back-derived, since dt is not fixed within a cohort for every
loader.

Output: results/rotation_power_bound.json -- per session: theta_se,
min_resolvable_theta_rad, min_resolvable_f_hz, observed_theta_rad,
observed_below_bound (whether the point estimate itself sits under the
session's own resolution floor); per cohort: min/median/max of the
resolvable bound across its sessions.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/run_rotation_power_bound.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_vstar_eigen_audit as vea  # noqa: E402  (ALL_ITERS -- dt per session, no refit)

RESULTS = ROOT / "results"
Z_ALPHA_HALF = 1.959963985  # z_{0.975}, two-sided alpha=0.05
Z_POWER = 1.644853627       # z_{0.95}, 95% power
DETECTION_FACTOR = Z_ALPHA_HALF + Z_POWER


def _session_dts() -> dict[str, dict[str, float]]:
    """{dataset: {session: dt}}, read directly off the same iterators the
    factorial script used -- cheap (data loading only, no DMD/bootstrap)."""
    dts: dict[str, dict[str, float]] = {}
    for it in vea.ALL_ITERS:
        for dataset, session, _Z_trials, dt, _r in it():
            dts.setdefault(dataset, {})[session] = float(dt)
    return dts


def main():
    factorial = json.load(open(RESULTS / "vstar_fit_selection_factorial.json"))
    print("Reading per-session dt from the same iterators as the factorial (no refit) ...")
    dts = _session_dts()

    out: dict[str, dict] = {}
    for dataset, sessions in factorial.items():
        if dataset == "_meta":
            continue
        out[dataset] = {}
        bounds_rad, bounds_hz = [], []
        for session, row in sessions.items():
            cell = row["ensemble"]["mod"]  # primary cell
            lo, hi = cell["theta_ci"]
            se_theta = (hi - lo) / (2 * Z_ALPHA_HALF)
            min_theta = se_theta * DETECTION_FACTOR
            dt = dts.get(dataset, {}).get(session)
            min_f_hz = min_theta / (2 * np.pi * dt) if dt else None
            observed_theta = abs(cell["theta"])
            entry = {
                "theta_se": float(se_theta),
                "min_resolvable_theta_rad": float(min_theta),
                "min_resolvable_f_hz": float(min_f_hz) if min_f_hz is not None else None,
                "observed_theta_rad": float(observed_theta),
                "observed_f_hz": float(cell["f_hz"]),
                "observed_below_resolution_floor": bool(observed_theta < min_theta),
                "phase_class": cell["phase_class"],
                "dt": dt,
            }
            out[dataset][session] = entry
            bounds_rad.append(min_theta)
            if min_f_hz is not None:
                bounds_hz.append(min_f_hz)
        n_below = sum(1 for e in out[dataset].values() if e["observed_below_resolution_floor"])
        n_tot = len(out[dataset])
        print(f"  {dataset}: min-resolvable |theta| (rad) across sessions -- "
              f"best={np.min(bounds_rad):.4f} median={np.median(bounds_rad):.4f} worst={np.max(bounds_rad):.4f}; "
              f"{n_below}/{n_tot} sessions have an observed |theta| BELOW their own resolution floor "
              "(cannot distinguish that session's rotation from zero)")
        out[dataset]["_cohort_summary"] = {
            "n_sessions": n_tot,
            "n_below_resolution_floor": n_below,
            "min_resolvable_theta_rad_best": float(np.min(bounds_rad)),
            "min_resolvable_theta_rad_median": float(np.median(bounds_rad)),
            "min_resolvable_theta_rad_worst": float(np.max(bounds_rad)),
            "min_resolvable_f_hz_best": float(np.min(bounds_hz)) if bounds_hz else None,
            "min_resolvable_f_hz_median": float(np.median(bounds_hz)) if bounds_hz else None,
            "min_resolvable_f_hz_worst": float(np.max(bounds_hz)) if bounds_hz else None,
        }

    with open(RESULTS / "rotation_power_bound.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/rotation_power_bound.json")


if __name__ == "__main__":
    main()
