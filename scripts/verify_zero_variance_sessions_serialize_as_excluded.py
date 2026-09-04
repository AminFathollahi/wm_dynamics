#!/usr/bin/env python3
"""Permanent regression guard on results/macaque_pfc_microstimulation_headline_robustness.json's
per_session block.

A prior version of that artifact's producer serialised sessions with no
within-session regressor variance the same way it serialised a genuine
fitted result: slope=0.0, slope_ci_lo=slope_ci_hi=0.0, p_value=1.0, with no
status field distinguishing "measured a zero" from "never fitted anything".
That was a divide-by-zero numerical guard's fallback output dressed as a
fitted slope. The producer has since been repaired to serialise those
sessions as excluded records -- null slope/ci/p, n=0, an explicit
`status: "excluded_no_regressor_variance"`, and a reason -- and every
genuinely fitted session carries `status: "fitted"` with real numbers.

This script asserts that repair holds, every time it is run:
  1. No session/arm entry claims `status: "fitted"` while also carrying the
     exact degenerate signature (slope 0.0, zero-width CI, p-value 1.0) --
     that signature returning on a "fitted" record would mean the guard's
     raw fallback output is being passed off as a measurement again.
  2. Every excluded entry carries null in all four estimate fields and a
     non-empty reason string.
  3. An independent reconstruction of each session's condition count --
     built directly from the pipeline's own build_session_features, not
     from the artifact under test -- agrees with the artifact's fitted vs.
     excluded split for every arm. This catches a session being excluded
     (or included) for the wrong reason, not just the right shape.
  4. Zero-drop accounting over sessions and arms reconciles.

Read-only with respect to results/macaque_pfc_microstimulation_headline_robustness.json: loaded
and never rewritten. The only output is
results/zero_variance_session_exclusion_guard.json.

Run:
    /home/amin/miniconda3/envs/wm_dynamics/bin/python scripts/verify_zero_variance_sessions_serialize_as_excluded.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_macaque_pfc_microstimulation_pipeline as pipeline  # noqa: E402

RESULTS = ROOT / "results"
ARMS = ("vstar_alignment", "min_energy_dir_alignment", "align_s_m2", "align_s_m3")
DEGENERATE_SIGNATURE = {"slope": 0.0, "slope_ci_lo": 0.0, "slope_ci_hi": 0.0, "p_value": 1.0}
ESTIMATE_FIELDS = ("slope", "slope_ci_lo", "slope_ci_hi", "p_value")


def _carries_degenerate_signature(entry: dict) -> bool:
    return all(entry.get(k) == v for k, v in DEGENERATE_SIGNATURE.items())


def _classify_sessions() -> dict:
    """Rebuild each session's cond_features via the SAME, unmodified
    build_session_features the delivered artifact's own producer script
    calls, and record why a session ends up with zero or one surviving
    stimulation condition -- the two ways the alignment-to-target modifier
    can carry no within-session variance for the per-session OLS slope."""
    per_session_cause = {}
    for prefix in pipeline.SESSIONS:
        feat = pipeline.build_session_features(prefix, structural_ctrl=None)
        if feat is None:
            per_session_cause[prefix] = {
                "n_cond_surviving_exclusion": 0, "n_rows": 0,
                "cause": "build_session_features_returned_none",
                "reason": "fewer than 10 control-condition trials available for the PCA/DMD fit",
            }
            continue
        n_cond = len(feat["cond_features"])
        n_rows = len(feat["rows"])
        if n_cond == 0:
            reason = (
                "every non-control stimulation condition in this session was dropped by the "
                "shorted-channel-electrode-survival check (build_session_features requires ALL "
                "stimulation electrodes of a condition to survive exclusion) -- zero usable rows, "
                "not zero effect"
            )
            cause = "zero_rows_no_surviving_condition"
        elif n_cond == 1:
            uniq = sorted({round(v["alignment_to_vstar"], 6) for v in feat["cond_features"].values()})
            reason = (
                f"exactly one stimulation condition survived the shorted-channel exclusion, so "
                f"alignment_to_vstar (and every other per-condition modifier) is a SINGLE constant "
                f"value ({uniq[0]:.5f}) broadcast to all {n_rows} rows in this session -- zero "
                "within-session regressor variance, not zero effect"
            )
            cause = "single_condition_zero_variance"
        else:
            reason = f"{n_cond} stimulation conditions survived exclusion -- genuine within-session variance"
            cause = "genuine_fit"
        per_session_cause[prefix] = {
            "n_cond_surviving_exclusion": n_cond, "n_rows": n_rows, "cause": cause, "reason": reason,
        }
    return per_session_cause


def main() -> None:
    with open(RESULTS / "macaque_pfc_microstimulation_headline_robustness.json") as f:
        robustness = json.load(f)
    per_session = robustness["per_session"]
    session_order = robustness["session_order"]
    n_sessions = robustness["n_sessions"]

    print("Reconstructing per-session condition counts (why a session has no within-session variance) ...")
    causes = _classify_sessions()
    predicted_excluded = sorted(
        s for s, c in causes.items() if c["cause"] in ("zero_rows_no_surviving_condition", "single_condition_zero_variance")
    )
    predicted_fitted = sorted(s for s, c in causes.items() if c["cause"] == "genuine_fit")

    failures = []
    per_session_reasons = {}
    delivered_fitted_by_arm = {arm: set() for arm in ARMS}
    delivered_excluded_by_arm = {arm: set() for arm in ARMS}

    for s in session_order:
        c = causes.get(s)
        if c is None:
            failures.append(f"{s}: appears in the artifact's session_order but not in pipeline.SESSIONS")
            continue
        per_session_reasons[s] = {
            "n_rows_in_delivered_artifact": per_session[s]["n_rows"],
            "n_cond_surviving_shorted_channel_exclusion": c["n_cond_surviving_exclusion"],
            "independently_reconstructed_cause": c["cause"],
            "independently_reconstructed_reason": c["reason"],
        }
        for arm in ARMS:
            entry = per_session[s][arm]
            status = entry.get("status")
            if status == "fitted":
                delivered_fitted_by_arm[arm].add(s)
                if _carries_degenerate_signature(entry):
                    failures.append(
                        f"{s}/{arm}: status='fitted' but carries the exact guard-fallback signature "
                        f"(slope 0.0, zero-width CI, p=1.0) -- the defect this guard exists to catch "
                        "has returned"
                    )
                for field in ESTIMATE_FIELDS:
                    if entry.get(field) is None:
                        failures.append(f"{s}/{arm}: status='fitted' but '{field}' is null")
                if not entry.get("n"):
                    failures.append(f"{s}/{arm}: status='fitted' but 'n' is missing or zero")
            elif status == "excluded_no_regressor_variance":
                delivered_excluded_by_arm[arm].add(s)
                for field in ESTIMATE_FIELDS:
                    if entry.get(field) is not None:
                        failures.append(f"{s}/{arm}: status='excluded_no_regressor_variance' but '{field}' is not null")
                # 'n' is the row count the session HAD, not necessarily zero: a
                # zero-rows exclusion carries n=0, but a single-surviving-condition
                # exclusion still has real rows whose modifier is simply constant --
                # the estimate fields being null is what marks the record excluded,
                # not the row count.
                if "n" not in entry or entry["n"] is None:
                    failures.append(f"{s}/{arm}: status='excluded_no_regressor_variance' but 'n' is missing")
                if not entry.get("reason"):
                    failures.append(f"{s}/{arm}: status='excluded_no_regressor_variance' but 'reason' is empty")
            else:
                failures.append(f"{s}/{arm}: unrecognized status {status!r}")

    cross_check_by_arm = {}
    for arm in ARMS:
        matches = delivered_fitted_by_arm[arm] == set(predicted_fitted) and delivered_excluded_by_arm[arm] == set(predicted_excluded)
        cross_check_by_arm[arm] = matches
        if not matches:
            failures.append(
                f"{arm}: independent reconstruction disagrees with the delivered fitted/excluded split "
                f"(predicted fitted={sorted(predicted_fitted)}, delivered fitted={sorted(delivered_fitted_by_arm[arm])})"
            )

    n_genuine = sum(1 for c in causes.values() if c["cause"] == "genuine_fit")
    n_zero_rows = sum(1 for c in causes.values() if c["cause"] == "zero_rows_no_surviving_condition")
    n_zero_var = sum(1 for c in causes.values() if c["cause"] == "single_condition_zero_variance")
    n_other = len(causes) - n_genuine - n_zero_rows - n_zero_var

    guard_ok = len(failures) == 0
    for f in failures:
        print(f"  FAIL: {f}")
    print(f"\nGuard result: {'PASS' if guard_ok else 'FAIL'} "
          f"({n_genuine} fitted, {n_zero_rows + n_zero_var} excluded, {n_sessions} sessions, "
          f"{len(ARMS)} arms each)")

    out = {
        "guard_target": "results/macaque_pfc_microstimulation_headline_robustness.json per_session block, every session x arm entry",
        "guard_description": (
            "asserts no 'fitted' entry carries the numerical guard's degenerate fallback signature "
            "(slope 0.0, zero-width CI, p=1.0), every excluded entry carries null estimates with a "
            "reason, and an independent reconstruction of each session's surviving-condition count "
            "agrees with the delivered fitted/excluded split"
        ),
        "guard_passed": guard_ok,
        "failures": failures,
        "per_session_reasons": per_session_reasons,
        "cross_check_matches_delivered_split_by_arm": cross_check_by_arm,
        "counts": {
            "n_sessions": n_sessions, "n_genuine_fit": n_genuine,
            "n_zero_rows": n_zero_rows, "n_single_condition_zero_variance": n_zero_var,
            "n_unclassified": n_other,
        },
        "zero_drop_accounting": {
            "sessions": {
                "seen": n_sessions, "examined": len(per_session_reasons), "skipped": n_sessions - len(per_session_reasons),
                "breakdown": {"genuine_fit": n_genuine, "zero_rows_no_surviving_condition": n_zero_rows,
                             "single_condition_zero_variance": n_zero_var},
            },
            "arms": {"seen": len(ARMS), "examined": len(ARMS), "skipped": 0, "list": list(ARMS)},
        },
    }

    def _json_safe(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    out_path = RESULTS / "zero_variance_session_exclusion_guard.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_json_safe)
    print(f"Saved {out_path}")

    assert guard_ok, f"{len(failures)} guard check(s) failed -- see 'failures' in {out_path}"


if __name__ == "__main__":
    main()
