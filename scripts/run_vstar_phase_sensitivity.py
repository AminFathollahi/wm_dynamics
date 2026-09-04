#!/usr/bin/env python3
"""Phase sensitivity of the macaque PFC microstimulation causal headline.

v_star is one particular direction within the real invariant plane of the
leading DMD mode: whenever that mode is a complex-conjugate pair (8/11
macaque PFC microstimulation sessions -- see results/vstar_eigen_audit.json), any direction within
the plane is an equally valid eigenvector up to the arbitrary phase
convention used to pick a single real vector. This script asks whether the
causal slope reported for vstar_alignment is a property of that whole plane,
or an accident of the particular phase convention used to extract v_star.

Two directly competing readings, adjudicated here:
  (a) PHASE-CARRIED AND FRAGILE -- re-phasing v* within the plane destroys
      the slope. The headline was conditional on a lucky arbitrary
      convention.
  (b) PLANE-CARRIED AND ROBUST -- re-phasing leaves the slope intact. Then
      the existing null on align_s_m2 (the whole plane, cluster p=0.781) is
      a power/metric artefact of scoring a 1-D direction against a 2-D
      subspace with a compressed, lower-variance regressor, not evidence
      that the plane carries no signal.

Method: reuse run_macaque_pfc_microstimulation_pipeline.build_session_features's fitted operator
and per-condition stimulation directions (via its geometry_out hook) so the
cluster-robust re-fit needs no PCA/DMD refit per draw -- only the alignment
scalar and the downstream slope/bootstrap change. The cross-fit pseudo-
outcome (phi) does not depend on the modifier, so it is computed once and
reused for every draw, exactly as run_macaque_pfc_microstimulation_headline_robustness.py already
does for its fixed set of arms.

Four direction choices, all scored against the same phi:
  (i)   v_star itself (the pre-registered, phase-canonicalized baseline).
  (ii)  K=50 random within-plane phases per session, redrawn per draw --
        reports the distribution of cluster-robust slopes and p-values.
  (iii) the plane's dominant real axis: the top eigenvector of the
        symmetrized (numerical-abscissa) operator restricted to the plane --
        a second principled, non-arbitrary choice distinct from v_star's
        phase convention.
  (iv)  one fixed random within-plane direction (a single draw, not the K=50
        distribution), reported as a standalone control arm.
A session whose leading mode is real has no phase freedom -- v_star is the
only candidate direction there, so it is held fixed across every draw.

Run (needs the macaque PFC microstimulation data mount):
    conda run -n wm_dynamics python scripts/run_vstar_phase_sensitivity.py
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

import run_macaque_pfc_microstimulation_pipeline as macaque_pfc_microstimulation  # noqa: E402
import run_macaque_pfc_microstimulation_headline_robustness as headline  # noqa: E402
from causal import benchmark_modifiers  # noqa: E402
from statistics import stable_seed  # noqa: E402

RESULTS = ROOT / "results"
K_DRAWS = 50
N_BOOT = 2000
COMPLEX_TOL = 1e-8


def _leading_mode_geometry(eigs: np.ndarray, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """(e1, e2, is_complex) for the leading (argmax Re(lambda)) mode: e1 is
    v_star itself; e2 (None if the mode is real) is the orthonormal in-plane
    complement, so any in-plane direction is cos(theta)*e1 + sin(theta)*e2."""
    order = np.argsort(eigs.real)[::-1]
    idx = order[0]
    lam = eigs[idx]
    w = vecs[:, idx]
    is_complex = abs(lam.imag) >= COMPLEX_TOL * (abs(lam.real) + 1e-12)
    e1 = macaque_pfc_microstimulation.canonicalize_eigenvector_phase(w)
    if not is_complex:
        return e1, None, False
    phase0 = np.angle(w[np.argmax(np.abs(w))])
    w_rot = w * np.exp(-1j * phase0)
    e2 = w_rot.imag - (w_rot.imag @ e1) * e1
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2, True


def _direction_at_theta(e1: np.ndarray, e2: np.ndarray | None, theta: float) -> np.ndarray:
    if e2 is None:
        return e1
    return np.cos(theta) * e1 + np.sin(theta) * e2


def _dominant_real_axis(A: np.ndarray, e1: np.ndarray, e2: np.ndarray | None) -> np.ndarray:
    """Top eigenvector of the symmetrized operator restricted to the plane
    (the numerical-abscissa direction of maximum instantaneous growth) -- a
    second principled in-plane direction, unrelated to eigenvector phase."""
    if e2 is None:
        return e1
    Q = np.column_stack([e1, e2])
    A2 = Q.T @ A @ Q
    S = (A2 + A2.T) / 2
    w, v = np.linalg.eigh(S)
    d = Q @ v[:, np.argmax(w)]
    return d / (np.linalg.norm(d) + 1e-12)


def _build_dataset():
    """Baseline row set (canonical v*) plus per-session geometry needed to
    re-score alignment against any other in-plane direction."""
    all_rows, session_order, geoms = [], [], []
    for prefix in macaque_pfc_microstimulation.SESSIONS:
        geom = {}
        try:
            feat = macaque_pfc_microstimulation.build_session_features(prefix, structural_ctrl=None, geometry_out=geom)
        except Exception as e:
            print(f"  {prefix} FAILED: {e}")
            feat = None
        if feat is None:
            print(f"  {prefix} SKIP (insufficient data)")
            continue
        si = len(session_order)
        session_order.append(prefix)
        geoms.append(geom)
        for row in feat["rows"]:
            row["session_idx"] = si
        all_rows.extend(feat["rows"])
    return all_rows, session_order, geoms


def _modifier_for_directions(all_rows: list[dict], session_idx: np.ndarray, cond_arr: np.ndarray,
                              geoms: list[dict], directions: list[np.ndarray]) -> np.ndarray:
    out = np.empty(len(all_rows))
    for s, geom in enumerate(geoms):
        mask = session_idx == s
        d = directions[s]
        cond_b_hat = geom["cond_B_hat"]
        out[mask] = [abs(float(cond_b_hat[c] @ d)) for c in cond_arr[mask]]
    return out


def _zscore(m: np.ndarray) -> np.ndarray:
    return (m - m.mean()) / m.std()


def _sweep(all_rows, session_idx, cond_arr, geoms, leading_geom, phi, n_sessions,
           row_mask: np.ndarray | None = None) -> dict:
    """K_DRAWS random within-plane phases, one fresh draw per session per
    draw; complex-mode sessions vary, real-mode sessions hold v_star fixed."""
    slopes, ps, survives = [], [], []
    for k in range(K_DRAWS):
        directions = []
        for s in range(n_sessions):
            e1, e2, is_complex = leading_geom[s]
            if is_complex:
                draw_rng = np.random.default_rng(stable_seed(f"phase_sensitivity_draw{k}_{s}"))
                theta = float(draw_rng.uniform(0, 2 * np.pi))
                directions.append(_direction_at_theta(e1, e2, theta))
            else:
                directions.append(e1)
        m = _modifier_for_directions(all_rows, session_idx, cond_arr, geoms, directions)
        if row_mask is not None:
            m, phi_use, sidx_use = m[row_mask], phi[row_mask], session_idx[row_mask]
        else:
            phi_use, sidx_use = phi, session_idx
        z = _zscore(m)
        boot_rng = np.random.default_rng(stable_seed(f"phase_sensitivity_boot{k}"))
        res = headline._cluster_robust_result(phi_use, z, sidx_use, n_sessions, boot_rng)
        slopes.append(res["slope"])
        ps.append(res["p_value"])
        survives.append(bool(res["ci_lo"] > 0 and res["p_value"] < 0.05))
    slopes = np.array(slopes)
    return {
        "k_draws": K_DRAWS,
        "median_slope": float(np.median(slopes)),
        "min_slope": float(slopes.min()),
        "max_slope": float(slopes.max()),
        "fraction_significant": float(np.mean(survives)),
        "all_slopes": slopes.tolist(),
        "all_p": [float(p) for p in ps],
    }


def _self_check() -> None:
    """Synthetic 2x2 rotation-decay block (complex pair) plus a distinct real
    eigenvalue, embedded in a 3x3 matrix: checks the plane basis is
    orthonormal, every in-plane direction is unit norm, and the dominant real
    axis has a strictly larger quadratic form than an arbitrary in-plane
    direction (the numerical-abscissa direction is a maximum by construction)."""
    rho, theta = 0.9, 0.4
    block = rho * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A = np.zeros((3, 3))
    A[:2, :2] = block
    A[2, 2] = 0.5
    eigs, vecs = np.linalg.eig(A)

    e1, e2, is_complex = _leading_mode_geometry(eigs, vecs)
    assert is_complex, "self-check: leading mode should be the complex-conjugate pair"
    assert abs(np.linalg.norm(e1) - 1.0) < 1e-10
    assert abs(np.linalg.norm(e2) - 1.0) < 1e-10
    assert abs(e1 @ e2) < 1e-10, "self-check: plane basis must be orthonormal"

    for th in np.linspace(0, 2 * np.pi, 9):
        d = _direction_at_theta(e1, e2, th)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-10

    d_dom = _dominant_real_axis(A, e1, e2)
    assert abs(np.linalg.norm(d_dom) - 1.0) < 1e-10
    quad_dom = float(d_dom @ ((A + A.T) / 2) @ d_dom)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d_rand = _direction_at_theta(e1, e2, float(rng.uniform(0, 2 * np.pi)))
        quad_rand = float(d_rand @ ((A + A.T) / 2) @ d_rand)
        assert quad_dom >= quad_rand - 1e-10, "self-check: dominant axis should maximize the quadratic form"

    # A purely real leading mode has no phase freedom: e2 is None, and
    # _direction_at_theta must ignore theta and return e1 unchanged.
    A_real = np.diag([0.9, 0.3, -0.2])
    eigs_r, vecs_r = np.linalg.eig(A_real)
    e1_r, e2_r, is_complex_r = _leading_mode_geometry(eigs_r, vecs_r)
    assert not is_complex_r and e2_r is None
    assert np.allclose(_direction_at_theta(e1_r, e2_r, 1.23), e1_r)
    print("Self-check passed (plane geometry, phase parameterization, dominant axis, real-mode no-freedom case).")


def main():
    _self_check()
    print("Building baseline row set (canonical v*) + per-session geometry ...")
    all_rows, session_order, geoms = _build_dataset()
    n_sessions = len(session_order)
    print(f"  {len(all_rows)} rows, {n_sessions} sessions")

    y = headline._col(all_rows, "y")
    t = headline._col(all_rows, "t").astype(int)
    vstar_raw = headline._col(all_rows, "modifier")
    propensity = headline._col(all_rows, "propensity")
    angle_idx = headline._col(all_rows, "angle_idx").astype(int)
    session_idx = headline._col(all_rows, "session_idx").astype(int)
    cond_arr = np.array([r["cond"] for r in all_rows], dtype=int)
    X = headline._build_X(all_rows, angle_idx, session_idx, n_sessions)

    leading_geom = [_leading_mode_geometry(g["eigs"], g["vecs"]) for g in geoms]
    is_complex = [lg[2] for lg in leading_geom]
    n_complex = sum(is_complex)
    complex_sessions = [session_order[s] for s in range(n_sessions) if is_complex[s]]
    real_sessions = [session_order[s] for s in range(n_sessions) if not is_complex[s]]
    print(f"  leading mode is a complex-conjugate pair (phase freedom) in {n_complex}/{n_sessions} sessions: "
          f"{complex_sessions}")
    print(f"  leading mode is real (no phase freedom -- clean cases) in {n_sessions - n_complex}/{n_sessions}: "
          f"{real_sessions}")

    print("\nCross-fitting the shared pseudo-outcome (phi; independent of the modifier) ...")
    bench_rng = np.random.default_rng(stable_seed("phase_sensitivity_phi"))
    bench = benchmark_modifiers(y, t, X, modifiers={"vstar_alignment": vstar_raw},
                                propensity=propensity, n_perm=50, rng=bench_rng)
    phi = bench["phi"]
    assert len(phi) == len(y), "benchmark_modifiers dropped rows -- alignment assumption violated"

    # (i) baseline: canonical v*, cluster-robust, self-checked against the
    # on-disk headline (fresh cross-fit -> ballpark match, not bit-identical).
    z_vstar = bench["z_modifiers"]["vstar_alignment"]
    base_rng = np.random.default_rng(stable_seed("phase_sensitivity_baseline"))
    baseline = headline._cluster_robust_result(phi, z_vstar, session_idx, n_sessions, base_rng)
    on_disk = json.load(open(RESULTS / "macaque_pfc_microstimulation_headline_robustness.json"))
    on_disk_vstar = on_disk["cluster_robust"]["vstar_alignment"]["cluster_robust"]
    print(f"\n(i) baseline v* : cluster-robust slope={baseline['slope']:+.4f} "
          f"[{baseline['ci_lo']:+.4f}, {baseline['ci_hi']:+.4f}] p={baseline['p_value']:.4f}  "
          f"(on-disk headline: slope={on_disk_vstar['slope']:+.4f} p={on_disk_vstar['p_value']:.4f})")

    # (ii) K=50 random within-plane phases -- pooled (all sessions) and
    # restricted to the sessions that actually have phase freedom.
    print(f"\n(ii) {K_DRAWS} random within-plane phase draws, pooled (N={n_sessions} sessions) ...")
    pooled_sweep = _sweep(all_rows, session_idx, cond_arr, geoms, leading_geom, phi, n_sessions)
    print(f"  median slope={pooled_sweep['median_slope']:+.4f}  "
          f"range=[{pooled_sweep['min_slope']:+.4f}, {pooled_sweep['max_slope']:+.4f}]  "
          f"fraction p<0.05={pooled_sweep['fraction_significant']:.2f}")

    complex_mask = np.isin(session_idx, [s for s in range(n_sessions) if is_complex[s]])
    print(f"\n(ii) same sweep restricted to the {n_complex} complex-mode sessions only "
          f"(N={int(complex_mask.sum())} rows) ...")
    complex_sweep = _sweep(all_rows, session_idx, cond_arr, geoms, leading_geom, phi, n_sessions,
                           row_mask=complex_mask)
    print(f"  median slope={complex_sweep['median_slope']:+.4f}  "
          f"range=[{complex_sweep['min_slope']:+.4f}, {complex_sweep['max_slope']:+.4f}]  "
          f"fraction p<0.05={complex_sweep['fraction_significant']:.2f}")

    # (iii) the plane's dominant real axis (numerical-abscissa direction).
    dom_directions = [_dominant_real_axis(g["A"], *leading_geom[s][:2]) for s, g in enumerate(geoms)]
    m_dom = _modifier_for_directions(all_rows, session_idx, cond_arr, geoms, dom_directions)
    dom_rng = np.random.default_rng(stable_seed("phase_sensitivity_dominant_axis"))
    dominant_axis_result = headline._cluster_robust_result(phi, _zscore(m_dom), session_idx, n_sessions, dom_rng)
    print(f"\n(iii) plane's dominant real axis: cluster-robust slope={dominant_axis_result['slope']:+.4f} "
          f"[{dominant_axis_result['ci_lo']:+.4f}, {dominant_axis_result['ci_hi']:+.4f}] "
          f"p={dominant_axis_result['p_value']:.4f}")

    # (iv) one fixed within-plane random direction (single draw, held control).
    single_directions = []
    for s in range(n_sessions):
        e1, e2, cplx = leading_geom[s]
        if cplx:
            rng_s = np.random.default_rng(stable_seed(f"phase_sensitivity_single_random_{session_order[s]}"))
            theta = float(rng_s.uniform(0, 2 * np.pi))
            single_directions.append(_direction_at_theta(e1, e2, theta))
        else:
            single_directions.append(e1)
    m_single = _modifier_for_directions(all_rows, session_idx, cond_arr, geoms, single_directions)
    single_rng = np.random.default_rng(stable_seed("phase_sensitivity_single_random_boot"))
    single_random_result = headline._cluster_robust_result(phi, _zscore(m_single), session_idx, n_sessions, single_rng)
    print(f"\n(iv) one within-plane random direction: cluster-robust slope={single_random_result['slope']:+.4f} "
          f"[{single_random_result['ci_lo']:+.4f}, {single_random_result['ci_hi']:+.4f}] "
          f"p={single_random_result['p_value']:.4f}")

    # Verdict: is the effect phase-carried-and-fragile, or plane-carried-and-robust?
    # Sign stability and significance stability are scored separately -- forcing
    # this onto a strict binary would overclaim in either direction if (as here)
    # the sign never flips but significance is not uniform across draws.
    baseline_survives = baseline["ci_lo"] > 0 and baseline["p_value"] < 0.05
    sign_never_flips = (pooled_sweep["min_slope"] > 0 and dominant_axis_result["slope"] > 0 and
                        single_random_result["slope"] > 0)
    majority_significant = pooled_sweep["fraction_significant"] >= 0.5
    if not baseline_survives:
        verdict = ("Baseline v* does not itself survive this fresh cross-fit cluster-robust re-fit "
                   "-- cannot adjudicate phase-sensitivity against an unstable reference; report as-is.")
    elif not sign_never_flips:
        verdict = (f"PHASE-CARRIED AND FRAGILE: across {K_DRAWS} random within-plane phase draws (min "
                   f"slope={pooled_sweep['min_slope']:+.4f}), the sign of the effect is not stable within the "
                   f"plane. The headline p=0.008 was conditional on the specific phase convention used to "
                   "extract v*, and this must be stated as a limitation in the abstract, not the discussion.")
    elif pooled_sweep["fraction_significant"] >= 0.8:
        verdict = (f"PLANE-CARRIED AND ROBUST: across {K_DRAWS} random within-plane phase draws, the "
                   f"cluster-robust slope stays positive and significant for {pooled_sweep['fraction_significant']:.0%} "
                   f"of draws (median={pooled_sweep['median_slope']:+.4f} vs baseline {baseline['slope']:+.4f}). "
                   "align_s_m2's null (cluster p=0.781) is a power/metric artefact of scoring a 1-D direction "
                   "against a 2-D subspace with a compressed, lower-variance regressor, not evidence that the "
                   "plane carries no additional causal signal.")
    else:
        verdict = (f"SIGN-STABLE, POWER-LIMITED IN SIGNIFICANCE (neither clean reading): across {K_DRAWS} random "
                   f"within-plane phase draws the slope NEVER changes sign (pooled range "
                   f"[{pooled_sweep['min_slope']:+.4f}, {pooled_sweep['max_slope']:+.4f}]; also positive for both "
                   f"the dominant real axis, slope={dominant_axis_result['slope']:+.4f} p={dominant_axis_result['p_value']:.4f}, "
                   f"and the single random in-plane draw, slope={single_random_result['slope']:+.4f} "
                   f"p={single_random_result['p_value']:.4f}), so this is NOT the phase-carried-and-fragile "
                   f"reading -- the plane genuinely carries a same-signed effect throughout. But only "
                   f"{pooled_sweep['fraction_significant']:.0%} of draws individually clear cluster-robust "
                   "p<0.05 (restricted to the 8 complex-mode sessions only, "
                   f"{complex_sweep['fraction_significant']:.0%}), so significance is power-limited at N=11 "
                   "session clusters rather than uniformly present. Reading: align_s_m2's null is PARTLY a "
                   "power/metric artefact (the sign is robust to phase) and partly genuine -- the specific "
                   "phase convention used for v* still matters for whether the effect clears significance in "
                   "any single draw, so v* is not interchangeable with an arbitrary in-plane direction on a "
                   "per-draw basis even though the plane as a whole is directionally consistent. Report both "
                   "halves; do not round this off to either clean case.")
    print(f"\nVERDICT: {verdict}")

    out = {
        "n_sessions": n_sessions,
        "session_order": session_order,
        "leading_mode_complex": {session_order[s]: bool(is_complex[s]) for s in range(n_sessions)},
        "n_complex_leading_mode": n_complex,
        "complex_sessions": complex_sessions,
        "real_sessions_no_phase_freedom": real_sessions,
        "baseline_vstar": baseline,
        "baseline_vs_on_disk": {"on_disk_slope": on_disk_vstar["slope"], "on_disk_p": on_disk_vstar["p_value"]},
        "random_phase_sweep_pooled": pooled_sweep,
        "random_phase_sweep_complex_sessions_only": complex_sweep,
        "dominant_real_axis": dominant_axis_result,
        "single_within_plane_random_direction": single_random_result,
        "verdict": verdict,
    }
    with open(RESULTS / "vstar_phase_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
    print("\nSaved results/vstar_phase_sensitivity.json")


if __name__ == "__main__":
    main()
