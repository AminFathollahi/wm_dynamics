#!/usr/bin/env python3
"""Hard gate for a phase x alignment CATE interaction (macaque PFC microstimulation only).

A session-specific stimulation phase phi(t_stim) is meaningless without an
identifiable rotational plane. This script applies the pre-registered gate
before considering any interaction model: a session qualifies only if (i)
the dominant eigenvalue pair is genuinely complex in the primary estimator
cell (ensemble + argmax|lambda|; theta bootstrap CI excludes zero -- read
directly from results/vstar_fit_selection_factorial.json, not recomputed)
AND (ii) the 2-D invariant plane is bootstrap-stable (mean bootstrap
subspace affinity at m=2 >= 0.9, the pre-registered threshold -- read
directly from results/vstar_subspace_stability.json's existing m=2
bootstrap affinities, not recomputed; this threshold matches the stable
regime (~0.996-1.000) already established for m=2 project-wide, contrasted
against the markedly less stable m=1 vector, ~0.67-0.79 on average).

Given this project's existing priors (a near-degenerate eigenvalue spectrum;
wide theta confidence intervals in most sessions), few or zero macaque PFC microstimulation
sessions are expected to pass. If fewer than 5 sessions qualify,
the interaction is NOT fit -- fitting it on an under-powered, cherry-picked
subset would manufacture a testable N rather than report the actual state of
the evidence. The gate threshold is stated before it is applied, and is not
adjusted after seeing how many sessions pass.

Output: results/rotation_phase_cate.json.

Run:
    conda run -n wm_dynamics python scripts/run_rotation_phase_cate.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
STABILITY_THRESHOLD = 0.9  # pre-registered: mean bootstrap subspace affinity at m=2
MIN_QUALIFYING = 5  # pre-registered: minimum sessions to fit the interaction at all


def theta_excludes_zero(theta_ci: list[float]) -> bool:
    return theta_ci[0] > 0 or theta_ci[1] < 0


def main():
    factorial = json.load(open(RESULTS / "vstar_fit_selection_factorial.json"))["macaque_pfc_microstimulation"]
    subspace = json.load(open(RESULTS / "vstar_subspace_stability.json"))

    per_session = {}
    for session, row in factorial.items():
        theta_ci = row["ensemble"]["mod"]["theta_ci"]
        complex_pair = theta_excludes_zero(theta_ci)
        plane_stability = subspace.get(session, {}).get("bootstrap_affinity_mean", {}).get("2")
        plane_stable = plane_stability is not None and plane_stability >= STABILITY_THRESHOLD
        per_session[session] = {
            "theta_ci": theta_ci, "complex_pair": complex_pair,
            "plane_bootstrap_affinity_m2": plane_stability, "plane_stable": plane_stable,
            "qualifies": bool(complex_pair and plane_stable),
        }
        print(f"  {session}: theta_ci={theta_ci} complex={complex_pair} "
              f"plane_affinity={plane_stability} stable={plane_stable} "
              f"-> {'QUALIFIES' if per_session[session]['qualifies'] else 'excluded'}")

    n_qualifying = sum(1 for v in per_session.values() if v["qualifies"])
    print(f"\nn_qualifying = {n_qualifying} (threshold: >= {MIN_QUALIFYING} sessions to fit the interaction)")

    if n_qualifying < MIN_QUALIFYING:
        out = {
            "status": "underpowered", "n_qualifying": n_qualifying, "n_total_sessions": len(per_session),
            "reason": (f"only {n_qualifying}/{len(per_session)} macaque PFC microstimulation sessions have both a genuinely "
                      f"complex dominant eigenvalue pair (theta CI excludes 0, primary cell) and a "
                      f"bootstrap-stable 2-D plane (mean affinity >= {STABILITY_THRESHOLD} at m=2) -- "
                      f"below the pre-registered minimum of {MIN_QUALIFYING} needed to fit a phase x "
                      f"alignment interaction without manufacturing a testable N from a cherry-picked "
                      f"subset. The interaction was NOT fit."),
            "per_session": per_session,
        }
        print(f"\nGATE FAILS: {out['reason']}")
    else:
        out = {"status": "gate_passed", "n_qualifying": n_qualifying, "per_session": per_session,
               "note": "interaction fit not implemented in this script -- see 31B for the nested "
                       "model comparison to run if this branch is ever reached"}
        print("\nGATE PASSES -- see module docstring's 31B for the interaction fit this would require.")

    with open(RESULTS / "rotation_phase_cate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved results/rotation_phase_cate.json")


if __name__ == "__main__":
    main()
