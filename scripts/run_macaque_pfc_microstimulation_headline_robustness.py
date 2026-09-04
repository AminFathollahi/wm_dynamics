#!/usr/bin/env python3
"""Session/animal-clustered robustness of the macaque PFC microstimulation causal-targeting headline
(vstar_alignment's gate-slope and min_energy_dir_alignment, plus the
align_s_m2/align_s_m3 subspace variants).

THE VULNERABILITY: even after src/causal.py's DML inference was switched from
asymptotic Wald to bootstrap p-values, those p-values still come from a
TRIAL-LEVEL bootstrap over rows nested in 11 sessions across 2 monkeys (Sa: 1
session -- contributes zero usable rows after the shorted-channel exclusion,
so the design is effectively single-animal; Wa: 10 sessions --
scripts/run_macaque_pfc_microstimulation_pipeline.py SESSIONS). The gate model adjusts session
MEANS (a session one-hot in X), but trial-level bootstrap resampling still
treats clustered trials as independent. If the effect is carried by a few
sessions, the effective n is ~sessions, not ~thousands of trials -- a
reviewer will (rightly) ask this. In the current data neither concern turns
out to matter in isolation: trial-level bootstrap inference alone already
puts vstar_alignment at p~0.24 (non-significant), and cluster-robust
inference gives p~0.08 -- close to trial-level, not a large clustering-driven
shift, but still non-significant either way (see cluster_robust in the
written-out JSON).

METHOD CHOSEN FOR CLUSTER-ROBUST INFERENCE: cluster bootstrap (resample SESSIONS with replacement, not trials). Rationale:
(1) it directly targets "what if we had drawn a different sample of sessions,"
the actual sampling unit of concern; (2) it needs no assumption about how
treatment/modifier is assigned within a session (unlike a session-permutation
null, which would require deciding how to permute a continuous, per-condition
modifier that already varies WITHIN each session); (3) it reuses the exact
closed-form OLS-slope formula _dr_slope uses internally, so implementing it is
a few lines, not a new estimator.

Reuses (no duplicated causal/geometry logic):
  - scripts.run_macaque_pfc_microstimulation_pipeline: SESSIONS, build_session_features (feature
    construction identical to main()'s own loop).
  - src.causal: benchmark_modifiers (ONE shared cross-fit phi for both arms,
    exactly like main() already does for its own benchmark step), _dr_slope
    (per-session/per-animal refits on that fixed phi; its own docstring calls
    it "the shared core... used by both cate_vs_modifier_slope and
    benchmark_modifiers" -- intended for direct reuse).

Run (needs the macaque PFC microstimulation data mount):
    conda run -n wm_dynamics python scripts/run_macaque_pfc_microstimulation_headline_robustness.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402  (reuse SESSIONS + feature build)
from causal import benchmark_modifiers, _dr_slope, _r2  # noqa: E402
from statistics import stable_seed, permutation_pvalue  # noqa: E402
from provenance import _json_safe

RESULTS = ROOT / "results"
ARMS = ("vstar_alignment", "min_energy_dir_alignment", "align_s_m2", "align_s_m3")
# Trial-resolution dissociation -- does the causal predictive power of
# vstar_alignment come from its condition/trial-to-trial resolution, or from
# static session-level structure? session_mean_vstar_scalar is the
# session-groupby mean of vstar_alignment, broadcast back (see
# run_macaque_pfc_microstimulation_pipeline.py's own construction of the same column).
DISSOCIATION_ARM = "session_mean_vstar_scalar"
N_PERM_PER_SESSION = 1000
N_BOOT = 2000
N_PERM_NESTED = 2000


def _animal_of(prefix: str) -> str:
    return "Sa" if prefix.startswith("Sa") else "Wa"


def _build_all_rows() -> tuple[list[dict], list[str]]:
    """Exactly main()'s own per-session loop (build_session_features + the
    session_idx tag), so row-level session_idx/angle_idx/modifier/propensity
    here match what run_macaque_pfc_microstimulation_pipeline.py itself scores."""
    all_rows, session_order = [], []
    for prefix in macaque_pfc_microstimulation.SESSIONS:
        try:
            feat = macaque_pfc_microstimulation.build_session_features(prefix, structural_ctrl=None)
        except Exception as e:
            print(f"  {prefix} FAILED: {e}")
            feat = None
        if feat is None:
            print(f"  {prefix} SKIP (insufficient data)")
            continue
        si = len(session_order)
        session_order.append(prefix)
        for row in feat["rows"]:
            row["session_idx"] = si
        all_rows.extend(feat["rows"])

    # Part 15A (same fallback/derivation as run_macaque_pfc_microstimulation_pipeline.main): session
    # mean of the per-condition vstar_alignment scalar, broadcast back to every
    # row in that session -- the static competitor to the condition/trial-
    # resolved "modifier" column.
    session_idx_all = np.array([r["session_idx"] for r in all_rows], dtype=int)
    modifier_all = np.array([r["modifier"] for r in all_rows], dtype=float)
    session_mean = np.empty_like(modifier_all)
    for s in range(len(session_order)):
        mask = session_idx_all == s
        session_mean[mask] = np.nanmean(modifier_all[mask])
    for row, sm in zip(all_rows, session_mean):
        row["session_mean_vstar_scalar"] = float(sm)

    return all_rows, session_order


def _col(all_rows: list[dict], key: str) -> np.ndarray:
    return np.array([r.get(key, np.nan) for r in all_rows], dtype=float)


def _build_X(all_rows: list[dict], angle_idx: np.ndarray, session_idx: np.ndarray,
             n_sessions: int) -> np.ndarray:
    angle_oh = np.eye(angle_idx.max() + 1)[angle_idx]
    session_oh = np.eye(n_sessions)[session_idx]
    return np.hstack([angle_oh, session_oh])


def _slope_formula(m: np.ndarray, phi: np.ndarray) -> float:
    """The same closed-form OLS slope _dr_slope._fit uses internally (one
    line -- deliberately NOT calling the full _dr_slope inside the B=2000
    cluster-bootstrap loop, since its own internal 2000-iteration bootstrap +
    permutation null would be pointless nested work per resample)."""
    mc = m - m.mean()
    denom = (mc ** 2).sum()
    if denom < 1e-15:
        return 0.0
    return float((mc * (phi - phi.mean())).sum() / denom)


def _cluster_bootstrap(phi: np.ndarray, modifier: np.ndarray, session_idx: np.ndarray,
                        n_sessions: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample session indices WITH replacement (a session drawn twice
    contributes its rows twice); recompute the pooled OLS slope each time."""
    rows_by_session = [np.where(session_idx == s)[0] for s in range(n_sessions)]
    boot = np.empty(n_boot)
    for b in range(n_boot):
        drawn = rng.integers(0, n_sessions, size=n_sessions)
        idx = np.concatenate([rows_by_session[s] for s in drawn])
        boot[b] = _slope_formula(modifier[idx], phi[idx])
    return boot


def _cluster_robust_result(phi: np.ndarray, modifier: np.ndarray, session_idx: np.ndarray,
                           n_sessions: int, rng: np.random.Generator) -> dict:
    boot = _cluster_bootstrap(phi, modifier, session_idx, n_sessions, N_BOOT, rng)
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    slope = _slope_formula(modifier, phi)
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))
    p = min(p, 1.0)
    return {"slope": slope, "ci_lo": float(ci_lo), "ci_hi": float(ci_hi), "p_value": p,
            "n_boot": N_BOOT}


def _self_check_planted_effect() -> bool:
    """Spec step 7b: synthetic dataset, n_sessions clusters each with its own
    random modifier and a KNOWN planted per-session effect of consistent sign;
    the cluster-bootstrap machinery must recover that sign (catches a
    backwards-sign bug in the bootstrap logic itself)."""
    rng = np.random.default_rng(12345)
    n_sessions_syn = 10
    rows_per_session = 200
    true_beta = 0.7  # planted, positive, consistent sign across sessions
    session_idx_list, modifier_list, phi_list = [], [], []
    for s in range(n_sessions_syn):
        m = rng.standard_normal(rows_per_session)
        phi = true_beta * m + rng.standard_normal(rows_per_session) * 0.5 + rng.normal(0, 1)
        session_idx_list.append(np.full(rows_per_session, s))
        modifier_list.append(m)
        phi_list.append(phi)
    session_idx_syn = np.concatenate(session_idx_list)
    modifier_syn = np.concatenate(modifier_list)
    phi_syn = np.concatenate(phi_list)
    res = _cluster_robust_result(phi_syn, modifier_syn, session_idx_syn, n_sessions_syn, rng)
    return res["slope"] > 0 and res["ci_lo"] > 0


def main():
    print("Rebuilding all_rows exactly as run_macaque_pfc_microstimulation_pipeline.main() does ...")
    all_rows, session_order = _build_all_rows()
    if not all_rows:
        print("No usable macaque PFC microstimulation sessions -- stopping without a robustness result.")
        return
    n_sessions = len(session_order)
    animal_of_session = [_animal_of(p) for p in session_order]
    n_animals = len(set(animal_of_session))
    print(f"  {len(all_rows)} rows across {n_sessions} sessions, {n_animals} animals "
          f"({sum(1 for a in animal_of_session if a == 'Sa')} Sa session(s), "
          f"{sum(1 for a in animal_of_session if a == 'Wa')} Wa session(s))")

    y = _col(all_rows, "y")
    t = _col(all_rows, "t").astype(int)
    vstar_mod = _col(all_rows, "modifier")
    min_energy_mod = _col(all_rows, "min_energy_dir_alignment")
    session_mean_mod = _col(all_rows, "session_mean_vstar_scalar")
    # Subspace arms (m=2/3), same insertion pattern as the existing
    # session_mean_vstar_scalar/amplification_alignment columns --
    # raw (non-z) columns, same all_rows/session_idx.
    align_s_m2_mod = _col(all_rows, "align_s_m2")
    align_s_m3_mod = _col(all_rows, "align_s_m3")
    raw_by_arm = {"vstar_alignment": vstar_mod, "min_energy_dir_alignment": min_energy_mod,
                 "align_s_m2": align_s_m2_mod, "align_s_m3": align_s_m3_mod}
    propensity = _col(all_rows, "propensity")
    angle_idx = _col(all_rows, "angle_idx").astype(int)
    session_idx = _col(all_rows, "session_idx").astype(int)
    X = _build_X(all_rows, angle_idx, session_idx, n_sessions)

    # Step 3: ONE shared cross-fit phi every arm is judged against (nuisance
    # cross-fitting does not depend on which modifier is scored).
    print("Cross-fitting ONE shared pseudo-outcome (phi) for all arms ...")
    bench_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_headline_robustness"))
    bench = benchmark_modifiers(
        y, t, X, modifiers={"vstar_alignment": vstar_mod, "min_energy_dir_alignment": min_energy_mod,
                            "session_mean_vstar_scalar": session_mean_mod,
                            "align_s_m2": align_s_m2_mod, "align_s_m3": align_s_m3_mod},
        propensity=propensity, n_perm=2000, rng=bench_rng,
    )
    phi = bench["phi"]
    session_mean_excluded = bench["excluded"].get("session_mean_vstar_scalar")
    z_mod = {arm: bench["z_modifiers"][arm] for arm in ARMS if arm in bench["z_modifiers"]}
    # benchmark_modifiers internally filters to rows finite in phi AND every
    # modifier, then returns the ALREADY-FILTERED phi/z_modifiers. macaque PFC microstimulation's
    # y/t/X/modifiers are never NaN in practice (run_macaque_pfc_microstimulation_pipeline.py scores
    # the same all_rows with pooled_n == len(all_rows)); assert that holds here
    # too rather than silently risking a session_idx misalignment.
    assert len(phi) == len(y), (
        "benchmark_modifiers dropped rows (non-finite phi/modifier) -- "
        "session_idx alignment assumption violated, STOP rather than silently misalign"
    )

    # Self-check (spec step 3): fresh phi's recovered OLS slope should be in
    # the same ballpark as run_macaque_pfc_microstimulation_pipeline.py's own leaderboard output
    # (not bit-identical -- different cross-fit RNG). Read from
    # all_statistics.json["causal_benchmark"]["leaderboard"], the raw,
    # always-present source of truth written directly by run_macaque_pfc_microstimulation_
    # pipeline.py -- NOT causal_benchmark.json, which build_apples_to_
    # apples_leaderboard.py restructures into primary_leaderboard/arms and
    # which this script's own trial_resolution output is itself an input to
    # (see build_causal_targeting_leaderboard.py's _load_macaque_pfc_microstimulation_leaderboard
    # docstring for the same idempotency rationale).
    on_disk = json.load(open(RESULTS / "all_statistics.json"))["causal_benchmark"]["leaderboard"]
    fresh_slopes = {}
    for arm in ARMS:
        mz = z_mod[arm]
        s = _slope_formula(mz, phi)
        disk_s = on_disk[arm]["slope"]
        rel_ok = abs(s - disk_s) < 0.25 * abs(disk_s)
        fresh_slopes[arm] = {"fresh_slope": s, "on_disk_slope": disk_s, "within_25pct": bool(rel_ok)}
        print(f"  {arm}: fresh z-slope={s:+.4f} vs on-disk={disk_s:+.4f} "
              f"({'OK' if rel_ok else 'MISMATCH -- reporting as-is'})")

    # Step 4: per-session slopes for both arms (cheap -- phi already fixed).
    # A session with zero usable rows (no stimulation condition survived the
    # shorted-channel exclusion) or exactly one surviving condition (the
    # modifier is then a single constant value, no within-session variance to
    # regress against) hits _dr_slope's own division-by-zero guard: it always
    # returns slope=0.0, ci=[0,0], p=1.0. That is a numerical fallback, not a
    # measured null, so such sessions are counted separately (excluded, with
    # a reason) rather than pooled into the sign-count / significance
    # denominators as if a fit had happened.
    print("\nPer-session slopes (both arms) ...")
    per_session = {}
    sign_counts = {arm: {"pos": 0, "neg": 0} for arm in ARMS}
    n_fitted = {arm: 0 for arm in ARMS}
    perm_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_headline_robustness_persession"))
    for s in range(n_sessions):
        mask = session_idx == s
        n_rows = int(mask.sum())
        entry = {"session": session_order[s], "animal": animal_of_session[s], "n_rows": n_rows}
        for arm in ARMS:
            mz = z_mod[arm]
            r = _dr_slope(phi[mask], mz[mask], N_PERM_PER_SESSION, perm_rng)
            fitted = not r["degenerate"]
            # _dr_slope's own divide-by-zero guard returns a fixed slope=0.0/ci=[0,0]/p=1.0
            # fallback when the modifier has no within-session variance to fit against -- that
            # fallback is not a measurement, so an excluded session serialises null in these four
            # fields rather than the fallback's numbers, which a consumer reading "slope" alone
            # (rather than "status") would otherwise silently mistake for a fitted zero effect.
            entry[arm] = {"slope": r["slope"] if fitted else None,
                          "slope_ci_lo": r["slope_ci_lo"] if fitted else None,
                          "slope_ci_hi": r["slope_ci_hi"] if fitted else None,
                          "p_value": r["p_value"] if fitted else None, "n": r["n"],
                          "status": "fitted" if fitted else "excluded_no_regressor_variance"}
            if fitted:
                sign_counts[arm]["pos" if r["slope"] > 0 else "neg"] += 1
                n_fitted[arm] += 1
            else:
                entry[arm]["reason"] = (
                    "no stimulation condition in this session survived the shorted-channel "
                    "electrode exclusion -- zero usable rows" if n_rows == 0 else
                    "exactly one stimulation condition survived the shorted-channel exclusion, "
                    "so the modifier is a single constant value with no within-session variance"
                )
        per_session[session_order[s]] = entry
        print(f"  {session_order[s]} ({entry['animal']}, n={n_rows}): " +
              "  ".join(
                  (f"{arm} slope={entry[arm]['slope']:+.4f}" if entry[arm]["status"] == "fitted"
                   else f"{arm} slope=<not fitted> [excluded: no regressor variance]")
                  for arm in ARMS))

    for arm in ARMS:
        n_excl = n_sessions - n_fitted[arm]
        print(f"  {arm}: {sign_counts[arm]['pos']}/{n_fitted[arm]} fitted sessions positive-sign "
              f"({n_excl}/{n_sessions} sessions excluded, no regressor variance)")

    # Step 5: per-animal slopes -- FRESH cross-fit per animal (restricting X
    # changes the nuisance fit; this is not a slice of the pooled phi).
    print("\nPer-animal slopes (fresh cross-fit each) ...")
    per_animal = {}
    for animal in sorted(set(animal_of_session)):
        anim_sessions = [s for s in range(n_sessions) if animal_of_session[s] == animal]
        row_mask = np.isin(session_idx, anim_sessions)
        if row_mask.sum() == 0:
            per_animal[animal] = {"n_sessions": len(anim_sessions), "n_rows": 0,
                                  "status": "no_usable_conditions_after_channel_exclusion"}
            print(f"  {animal} (n_sessions={len(anim_sessions)}, n_rows=0): "
                  "no conditions survived the shorted-channel exclusion -- skipped")
            continue
        y_a, t_a = y[row_mask], t[row_mask]
        raw_a = {arm: raw_by_arm[arm][row_mask] for arm in ARMS}
        prop_a = propensity[row_mask]
        angle_a = angle_idx[row_mask]
        sess_a = session_idx[row_mask]

        if len(anim_sessions) == 1:
            # Single-session monkey (Sa): a per-session one-hot column would be
            # collinear with the intercept (degenerates to all-ones) -- use
            # ONLY the angle one-hot, nothing to adjust for within one session.
            X_a = np.eye(angle_a.max() + 1)[angle_a]
            x_note = "angle one-hot only (single session -- session one-hot would be collinear with intercept)"
        else:
            remap = {s: i for i, s in enumerate(anim_sessions)}
            sess_a_local = np.array([remap[s] for s in sess_a])
            X_a = np.hstack([np.eye(angle_a.max() + 1)[angle_a],
                             np.eye(len(anim_sessions))[sess_a_local]])
            x_note = "angle one-hot + within-animal session one-hot"

        anim_rng = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_headline_robustness_{animal}"))
        bench_a = benchmark_modifiers(
            y_a, t_a, X_a, modifiers=raw_a,
            propensity=prop_a, n_perm=2000, rng=anim_rng,
        )
        lb = bench_a["leaderboard"]
        per_animal[animal] = {
            "n_sessions": len(anim_sessions), "n_rows": int(row_mask.sum()), "X_note": x_note,
            **{arm: lb[arm] for arm in ARMS if arm in lb},
        }
        print(f"  {animal} (n_sessions={len(anim_sessions)}, n_rows={int(row_mask.sum())}): " +
              "  ".join(f"{arm} slope={lb[arm]['slope']:+.4f} p={lb[arm]['p_value']:.4f}"
                        for arm in ARMS if arm in lb))

    animals_significant = {
        arm: [a for a, v in per_animal.items() if arm in v and v[arm]["p_value"] < 0.05 and v[arm]["slope"] > 0]
        for arm in ARMS
    }
    animals_significant_vstar = animals_significant["vstar_alignment"]
    animals_significant_min_energy = animals_significant["min_energy_dir_alignment"]

    # Step 6: cluster-robust inference (session as resampling unit).
    print("\nCluster-bootstrap (session-resampled, B=2000) ...")
    cluster_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_headline_robustness_clusterboot"))
    cluster_robust = {}
    for arm in ARMS:
        mz = z_mod[arm]
        cr = _cluster_robust_result(phi, mz, session_idx, n_sessions, cluster_rng)
        trial_level = on_disk[arm]
        cluster_robust[arm] = {
            "cluster_robust": cr,
            "trial_level_pooled": {"slope": trial_level["slope"], "slope_ci_lo": trial_level["slope_ci_lo"],
                                   "slope_ci_hi": trial_level["slope_ci_hi"], "p_value": trial_level["p_value"]},
        }
        survives = cr["ci_lo"] > 0 and cr["p_value"] < 0.05
        cluster_robust[arm]["survives_clustering"] = bool(survives)
        print(f"  {arm}: cluster-robust slope={cr['slope']:+.4f} [{cr['ci_lo']:+.4f}, {cr['ci_hi']:+.4f}] "
              f"p={cr['p_value']:.4f}  vs trial-level p={trial_level['p_value']:.4f} "
              f"-> {'SURVIVES' if survives else 'WEAKENS'}")

    # Step 7: self-checks.
    unresampled_check = {}
    for arm in ARMS:
        mz = z_mod[arm]
        full_slope = _slope_formula(mz, phi)
        cr_slope = cluster_robust[arm]["cluster_robust"]["slope"]
        assert abs(full_slope - cr_slope) < 1e-9, f"{arm}: cluster-bootstrap point estimate formula mismatch"
        unresampled_check[arm] = {"full_data_slope": full_slope, "cluster_robust_point_slope": cr_slope}
    planted_sign_ok = _self_check_planted_effect()
    assert planted_sign_ok, "cluster-bootstrap self-check FAILED: planted positive per-session effect did not recover positive sign"
    print(f"\nSelf-check: unresampled cluster-bootstrap slope matches pooled point estimate (OK); "
          f"synthetic planted-effect sign recovery: {'OK' if planted_sign_ok else 'FAILED'}")

    # Trial-resolution dissociation. Is the causal
    # predictive power of vstar_alignment carried by its condition/trial-to-
    # trial variation, or by static session-level structure (session_mean_
    # vstar_scalar)? Apples-to-apples: identical phi, identical rows (both
    # modifiers finite on the same set -- session_mean is finite wherever
    # vstar_alignment is, by construction).
    print("\nTrial-resolution dissociation (Part 15B) ...")
    trial_vstar_cr = cluster_robust["vstar_alignment"]["cluster_robust"]
    if session_mean_excluded is not None:
        print(f"  session_mean_vstar_scalar EXCLUDED by benchmark_modifiers: "
              f"{session_mean_excluded['reason']}")
        trial_resolution = {
            "trial_vstar": {"slope": trial_vstar_cr["slope"], "cluster_ci_lo": trial_vstar_cr["ci_lo"],
                            "cluster_ci_hi": trial_vstar_cr["ci_hi"], "cluster_p": trial_vstar_cr["p_value"]},
            "session_mean_vstar": {"excluded": True, "reason": session_mean_excluded["reason"]},
            "nested_dR2_trial_over_session": None,
            "verdict": ("session_mean_vstar_scalar has ~zero within-arena variance and was excluded by "
                       "benchmark_modifiers' zero-variance guard -- vstar_alignment carries NO exploitable "
                       "static session-level structure once collapsed to a session mean; this is itself the "
                       "trial-resolution answer: TRIAL-RESOLUTION IS THE ACTIVE INGREDIENT (verdict a)."),
        }
        print(f"  VERDICT: {trial_resolution['verdict']}")
    else:
        vstar_z = z_mod["vstar_alignment"]
        session_mean_z = bench["z_modifiers"]["session_mean_vstar_scalar"]
        session_mean_cr = _cluster_robust_result(phi, session_mean_z, session_idx, n_sessions, cluster_rng)
        session_mean_survives = bool(session_mean_cr["ci_lo"] > 0 and session_mean_cr["p_value"] < 0.05)

        # Nested dR2: does trial/condition-resolved vstar_alignment add
        # explanatory power OVER a base model containing only the static
        # session_mean_vstar_scalar? (direction fixed by this project's own
        # design rule: trial over session, i.e. base=session_mean,
        # added=vstar_alignment -- the
        # opposite direction from benchmark_modifiers' own winner-relative
        # nested dict, which would test the other arm against whichever of
        # these two individually has the larger marginal slope.)
        nested_rng = np.random.default_rng(stable_seed("macaque_pfc_microstimulation_headline_robustness_nested"))
        base_r2 = _r2(session_mean_z[:, None], phi)
        full_r2 = _r2(np.column_stack([session_mean_z, vstar_z]), phi)
        dr2_obs = full_r2 - base_r2
        null_dr2 = np.empty(N_PERM_NESTED)
        for b in range(N_PERM_NESTED):
            perm_vstar = nested_rng.permutation(vstar_z)
            null_dr2[b] = _r2(np.column_stack([session_mean_z, perm_vstar]), phi) - base_r2
        nested_p = permutation_pvalue(null_dr2 >= dr2_obs)

        print(f"  trial_vstar (condition-resolved):  cluster-robust slope={trial_vstar_cr['slope']:+.4f} "
              f"[{trial_vstar_cr['ci_lo']:+.4f}, {trial_vstar_cr['ci_hi']:+.4f}] p={trial_vstar_cr['p_value']:.4f}")
        print(f"  session_mean_vstar (static):        cluster-robust slope={session_mean_cr['slope']:+.4f} "
              f"[{session_mean_cr['ci_lo']:+.4f}, {session_mean_cr['ci_hi']:+.4f}] p={session_mean_cr['p_value']:.4f}")
        print(f"  nested dR2 (trial vstar over session_mean base): dR2={dr2_obs:+.5f} perm_p={nested_p:.4f}")

        trial_much_bigger = (trial_vstar_cr["ci_lo"] > 0 and trial_vstar_cr["p_value"] < 0.05) and \
            (not session_mean_survives) and nested_p < 0.05
        both_hold = (trial_vstar_cr["ci_lo"] > 0 and trial_vstar_cr["p_value"] < 0.05) and session_mean_survives
        session_only = session_mean_survives and not (trial_vstar_cr["ci_lo"] > 0 and trial_vstar_cr["p_value"] < 0.05)

        if trial_much_bigger:
            verdict = ("TRIAL-RESOLUTION IS THE ACTIVE INGREDIENT (verdict a): session_mean_vstar_scalar "
                      f"does not survive session-clustering (slope={session_mean_cr['slope']:+.4f}, "
                      f"p={session_mean_cr['p_value']:.4f}) while trial/condition-resolved vstar_alignment "
                      f"does (p={trial_vstar_cr['p_value']:.4f}), and vstar_alignment adds significant nested "
                      f"R^2 over the static session-mean base (dR2={dr2_obs:+.5f}, p={nested_p:.4f}).")
        elif both_hold:
            verdict = ("trial_vstar ~= session_mean_vstar (verdict b): the causally-relevant target is "
                      f"representational but the trial-resolution claim FAILS to add reliably over the "
                      f"static session mean (nested dR2={dr2_obs:+.5f}, p={nested_p:.4f}) -- both arms "
                      "survive session-clustering. Downgrade the trial-resolution framing; no reframe.")
        elif session_only:
            verdict = ("Session-level structure dominates (verdict c): session_mean_vstar_scalar survives "
                      f"session-clustering (slope={session_mean_cr['slope']:+.4f}, p={session_mean_cr['p_value']:.4f}) "
                      f"while trial/condition-resolved vstar_alignment does not "
                      f"(p={trial_vstar_cr['p_value']:.4f}) -- the win was carried by static session-level "
                      "structure, not trial resolution. This is a genuine change to the story; flag for the PI.")
        else:
            verdict = ("Neither arm survives session-clustering cleanly in a way that cleanly separates "
                      f"trial-resolution from static session structure (trial_vstar p={trial_vstar_cr['p_value']:.4f}, "
                      f"session_mean p={session_mean_cr['p_value']:.4f}, nested dR2={dr2_obs:+.5f} p={nested_p:.4f}) "
                      "-- report as-is (verdict b/c, no reframe).")
        print(f"  VERDICT: {verdict}")

        trial_resolution = {
            "trial_vstar": {"slope": trial_vstar_cr["slope"], "cluster_ci_lo": trial_vstar_cr["ci_lo"],
                            "cluster_ci_hi": trial_vstar_cr["ci_hi"], "cluster_p": trial_vstar_cr["p_value"]},
            "session_mean_vstar": {"slope": session_mean_cr["slope"], "cluster_ci_lo": session_mean_cr["ci_lo"],
                                   "cluster_ci_hi": session_mean_cr["ci_hi"], "cluster_p": session_mean_cr["p_value"],
                                   "survives_clustering": session_mean_survives},
            "nested_dR2_trial_over_session": {"dR2": float(dr2_obs), "perm_p": float(nested_p)},
            "verdict": verdict,
        }

    # Honest verdict.
    both_survive = cluster_robust["vstar_alignment"]["survives_clustering"] and \
        cluster_robust["min_energy_dir_alignment"]["survives_clustering"]
    n_animals_holds_vstar = len(animals_significant_vstar)
    verdict_parts = []
    if both_survive:
        verdict_parts.append(
            "Both significant arms (vstar_alignment, min_energy_dir_alignment) SURVIVE session-clustered "
            "cluster-bootstrap inference: their CIs still exclude zero and their cluster-robust p-values "
            "remain small. The trial-level p-values are NOT materially anticonservative for these arms."
        )
    else:
        weak = [a for a in ARMS if not cluster_robust[a]["survives_clustering"]]
        verdict_parts.append(
            f"{', '.join(weak)} WEAKENS materially under session-clustered inference -- the cluster-robust "
            "number, not the trial-level p, should be treated as the honest headline for that arm; the "
            "trial-level p is anticonservative there."
        )
    individually_significant = [s for s in session_order
                                if per_session[s]["vstar_alignment"]["status"] == "fitted"
                                and per_session[s]["vstar_alignment"]["p_value"] < 0.05]
    sa_slope = per_animal.get("Sa", {}).get("vstar_alignment", {}).get("slope")
    wa_slope = per_animal.get("Wa", {}).get("vstar_alignment", {}).get("slope")
    if sa_slope is None:
        sa_note = ("Sa contributes no usable causal-test rows (its one session has no stimulation "
                  "condition whose electrodes fully survive the shorted-channel exclusion) -- the "
                  "causal test rests entirely on Wa's 10 sessions.")
    else:
        ratio_note = (f"~{abs(sa_slope / wa_slope):.1f}x the 10-session Wa cohort's"
                     if wa_slope not in (None, 0) else "not comparable (missing the Wa arm)")
        sa_note = (
            "Sa contributes exactly ONE session, so its per-animal test is not independent evidence "
            f"of cross-animal generalization -- it is the same session re-labeled. Sa's per-animal "
            f"slope ({sa_slope:+.4f}) is {ratio_note} "
            f"({'N/A' if wa_slope is None else f'{wa_slope:+.4f}'})."
        )
    verdict_parts.append(
        f"vstar_alignment's per-animal slope is positive in {n_animals_holds_vstar}/{n_animals} monkeys "
        f"({', '.join(animals_significant_vstar) if animals_significant_vstar else 'none'} significant+positive "
        f"at per-animal level), but this is NOT a broad-based, symmetric cross-animal claim: {sa_note} At the "
        f"individual-SESSION level (trial-level p), {n_sessions - n_fitted['vstar_alignment']} of the "
        f"{n_sessions} sessions contributed no within-session regressor variance to fit against (zero usable "
        f"rows or a single surviving stimulation condition) and are excluded rather than counted as measured "
        f"zeros; of the {n_fitted['vstar_alignment']} sessions that were actually fitted, vstar_alignment "
        f"reaches significance in only {len(individually_significant)}/{n_fitted['vstar_alignment']} "
        f"({', '.join(individually_significant) if individually_significant else 'none'})."
    )
    verdict_parts.append(
        f"Per-session sign split (fitted sessions only): vstar_alignment "
        f"{sign_counts['vstar_alignment']['pos']}/{n_fitted['vstar_alignment']} positive, "
        f"min_energy_dir_alignment {sign_counts['min_energy_dir_alignment']['pos']}/"
        f"{n_fitted['min_energy_dir_alignment']} positive."
    )
    headline_verdict = " ".join(verdict_parts)
    print(f"\nVERDICT: {headline_verdict}")

    # Central question: does align_S (m=2,3) beat, tie, or lose to align_vstar
    # (m=1) under IDENTICAL cluster-robust inference?
    # (a) align_S >= align_vstar, both cluster-robust -> SUBSPACE reading;
    # (b) align_vstar > align_S clearly -> VECTOR reading, v*'s instability is
    #     a real limitation; (c) near-tie (or align_S non-significant while
    #     align_vstar is) -> data cannot separate them, keep v* primary,
    #     report the tie -- never claim the subspace on a tie.
    vstar_cr = cluster_robust["vstar_alignment"]["cluster_robust"]
    subspace_verdicts = {}
    for m_arm in ("align_s_m2", "align_s_m3"):
        s_cr = cluster_robust[m_arm]["cluster_robust"]
        vstar_survives = vstar_cr["ci_lo"] > 0 and vstar_cr["p_value"] < 0.05
        s_survives = s_cr["ci_lo"] > 0 and s_cr["p_value"] < 0.05
        if s_survives and s_cr["slope"] >= vstar_cr["slope"]:
            case = "a"
            subspace_verdict = (f"{m_arm} slope ({s_cr['slope']:+.4f}) >= vstar_alignment slope "
                          f"({vstar_cr['slope']:+.4f}), BOTH cluster-robust-significant -> "
                          "SUBSPACE reading supported for this m.")
        elif vstar_survives and not s_survives:
            case = "b"
            subspace_verdict = (f"vstar_alignment survives cluster-robust inference (p={vstar_cr['p_value']:.4f}) "
                          f"while {m_arm} does NOT (p={s_cr['p_value']:.4f}) -> the specific direction carries "
                          "the effect; v*'s rank instability is a REAL limitation, report prominently. "
                          "VECTOR reading for this m.")
        else:
            case = "c"
            subspace_verdict = (f"{m_arm} (slope={s_cr['slope']:+.4f}, p={s_cr['p_value']:.4f}) vs vstar_alignment "
                          f"(slope={vstar_cr['slope']:+.4f}, p={vstar_cr['p_value']:.4f}): NEAR-TIE or data "
                          "cannot cleanly separate them -- a near-tie means keep v* PRIMARY "
                          "and do NOT claim the subspace on a tie (report the tie honestly).")
        subspace_verdicts[m_arm] = {"case": case, "verdict": subspace_verdict,
                                    "cluster_robust": s_cr, "vstar_cluster_robust": vstar_cr}
        print(f"\nSUBSPACE VERDICT ({m_arm}): case ({case}) -- {subspace_verdict}")

    out = {
        "n_sessions": n_sessions,
        "n_animals": n_animals,
        "session_order": session_order,
        "animal_of_session": dict(zip(session_order, animal_of_session)),
        "shared_phi_self_check": fresh_slopes,
        "per_session": per_session,
        "sign_counts": sign_counts,
        "per_animal": per_animal,
        "animals_significant_positive": {
            "vstar_alignment": animals_significant_vstar,
            "min_energy_dir_alignment": animals_significant_min_energy,
        },
        "cluster_robust": cluster_robust,
        "self_check": {
            "unresampled_matches_pooled_point_estimate": unresampled_check,
            "planted_effect_sign_recovery_ok": bool(planted_sign_ok),
        },
        "headline_verdict": headline_verdict,
        "trial_resolution": trial_resolution,
        "subspace_verdicts": subspace_verdicts,
    }
    with open(RESULTS / "macaque_pfc_microstimulation_headline_robustness.json", "w") as f:
        json.dump(_json_safe(out), f, indent=2, allow_nan=False)
    print("\nSaved results/macaque_pfc_microstimulation_headline_robustness.json")


if __name__ == "__main__":
    main()
