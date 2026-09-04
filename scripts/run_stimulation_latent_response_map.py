#!/usr/bin/env python3
"""One estimator for the stimulation-evoked displacement of the latent
neural state, applied identically to every stimulation arm this project
holds, in a unit that is comparable across sessions and participants.

Four arms hold stimulation delivered while neural activity was recorded,
already analysed by this project with different machinery, in different
units, against different references. No statement currently exists about
what stimulation does to the latent state as such: an existing human
open-loop estimand records its own limitation explicitly -- a raw RMS
PC1-deviation number, in PC-score units, not cross-session-comparable. This
script repairs that: one displacement statistic, normalised by each
session's or participant's own spontaneous (unstimulated) state variability
in the same latent frame, laid side by side across arms.

This is explicitly NOT an epoch, species, or modality contrast. The four
arms differ in species, recording modality, task, stimulation waveform,
stimulation site and stimulated epoch simultaneously. No cross-arm
difference is attributed to any single one of those factors; a mandatory
confound table accompanies every cross-arm number.

Nothing already delivered is modified, re-run in place, or re-labelled.
Every existing fired branch and every existing recorded verdict stands and
is carried forward verbatim -- in particular the macaque delay-period
causal-gate verdict (FAIL/WEAK) and the classifier-triggered human arm's
item-level descriptive-only (never causal) restriction.

Output: results/stimulation_latent_response_map.json

Run:
    conda run -n wm_dynamics python scripts/run_stimulation_latent_response_map.py
"""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import importlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from control import (  # noqa: E402
    dominant_eigenmode,
    invariant_subspace_basis,
    stimulation_input_alignment,
    subspace_alignment,
)
from dynamics import fit_retention_dynamics  # noqa: E402
from geometry import pca_decompose  # noqa: E402
from provenance import canonical_json, git_commit  # noqa: E402
from statistics import (  # noqa: E402
    minimum_detectable_paired_difference,
    paired_sign_flip_test,
    pearson_permutation_test,
    permutation_pvalue,
    stable_seed,
)

RESULTS = ROOT / "results"
CHECKPOINT_DIR = RESULTS / ".checkpoints" / "run_stimulation_latent_response_map"
DEFAULT_N_NULL_DRAWS = 1000
DEFAULT_N_ROTATION_NULL = 1000
ALPHA = 0.05


# ── Checkpointing (fit-level, crash-proof) ──────────────────────────────────────

def _checkpoint_path(unit: str) -> Path:
    safe = unit.replace("/", "_")
    return CHECKPOINT_DIR / f"{safe}.json"


def load_checkpoint(unit: str) -> dict | None:
    """An unparseable or incomplete checkpoint record is treated as absent."""
    path = _checkpoint_path(unit)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict) or data.get("_complete") is not True:
        return None
    return data["record"]


def save_checkpoint(unit: str, record: dict) -> None:
    """Write to a temp file, then os.replace -- the completion flag is only
    ever written as part of the same atomic replace, after the fit that
    computed `record` has already returned, so a killed process never leaves
    a checkpoint that reads as complete but isn't."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(unit)
    payload = {"_complete": True, "record": record}
    fd, tmp_name = tempfile.mkstemp(dir=str(CHECKPOINT_DIR), prefix="._tmp_")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(canonical_json(payload))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def run_checkpointed(unit: str, fit_fn):
    """Load a cached, complete record for `unit`, or call `fit_fn()` (which
    must return a JSON-safe dict) and checkpoint the result before returning
    it. `fit_fn` is only ever invoked when no valid checkpoint exists."""
    cached = load_checkpoint(unit)
    if cached is not None:
        return cached
    record = fit_fn()
    save_checkpoint(unit, record)
    return record


# ── Common estimator primitives ─────────────────────────────────────────────────

def per_trial_state_norm(scores: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Per-trial scalar distance from a fixed centroid, in a latent frame
    fit on unstimulated trials: mean, over within-trial timepoints, of the
    per-timepoint Euclidean norm of the deviation from that centroid (norm
    taken before averaging over time, so an oscillatory, zero-mean-over-time
    signal does not trivially cancel to zero). One scalar per trial.

    scores   : (n_trials, n_bins, k) latent trajectories already projected
               into the frame (k=1 is a valid, degenerate case).
    centroid : (k,) reference point, typically the unstimulated population's
               own mean latent position in the same frame.
    """
    return np.linalg.norm(scores - centroid[None, None, :], axis=-1).mean(axis=1)


def displacement_and_normalisation(state: np.ndarray, unstim_mask: np.ndarray,
                                    stim_mask: np.ndarray) -> dict:
    """The common displacement statistic: the stimulated population's mean
    distance from the unstimulated centroid, net of the unstimulated
    population's own inherent spread from that same centroid (raw units),
    divided by the standard deviation of the unstimulated trial-to-trial
    state in the same frame (this leg's own normalisation -- the point of
    this analysis, and what makes the raw units comparable across arms).
    """
    unstim_state, stim_state = state[unstim_mask], state[stim_mask]
    if len(unstim_state) < 3 or len(stim_state) < 1:
        return {"status": "not_estimable",
                "reason": f"{len(unstim_state)} unstimulated and {len(stim_state)} stimulated trials"}
    baseline_dispersion = float(unstim_state.mean())
    condition_dispersion = float(stim_state.mean())
    spontaneous_sd = float(unstim_state.std(ddof=1))
    raw_displacement = condition_dispersion - baseline_dispersion
    normalized = raw_displacement / spontaneous_sd if spontaneous_sd > 0 else None
    return {
        "status": "complete",
        "n_unstim_trials": int(len(unstim_state)), "n_stim_trials": int(len(stim_state)),
        "baseline_dispersion_raw": baseline_dispersion, "condition_dispersion_raw": condition_dispersion,
        "raw_displacement": raw_displacement, "spontaneous_sd_raw": spontaneous_sd,
        "normalized_displacement": normalized,
    }


def within_unit_permutation_null(state_fn, n_unstim: int, n_stim: int, n_total: int,
                                  n_draws: int, rng: np.random.Generator) -> np.ndarray | None:
    """>=1000-draw null for a displacement statistic: the stimulation label
    is permuted within the session/participant's own trial pool (preserving
    the true unstim/stim group sizes) and the WHOLE statistic -- frame refit,
    displacement, normalisation -- is recomputed end to end from that fake
    label, exactly mirroring what a genuinely null (no true stimulation
    effect) world would produce under this same trial pool and estimator.

    `state_fn(unstim_mask, stim_mask) -> normalized_displacement or None`
    does the refit; this function only handles the label permutation.
    """
    if n_unstim < 3 or n_stim < 1:
        return None
    draws = np.full(n_draws, np.nan)
    for d in range(n_draws):
        perm = rng.permutation(n_total)
        fake_unstim = np.zeros(n_total, dtype=bool)
        fake_unstim[perm[:n_unstim]] = True
        fake_stim = np.zeros(n_total, dtype=bool)
        fake_stim[perm[n_unstim:n_unstim + n_stim]] = True
        value = state_fn(fake_unstim, fake_stim)
        draws[d] = value if value is not None else np.nan
    return draws


def rotation_null_alignment(basis: np.ndarray, direction: np.ndarray, n_draws: int,
                             rng: np.random.Generator) -> dict:
    """Alignment of `direction` (unit vector) to the subspace spanned by
    `basis`'s columns (src.control.subspace_alignment, reused, not
    re-derived), against a null of the same statistic on random unit
    directions in the same latent space -- what a fully-unaligned direction
    would produce by chance."""
    k = basis.shape[0]
    observed = subspace_alignment(basis, direction)
    random_dirs = rng.standard_normal((n_draws, k))
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-12
    null = np.array([subspace_alignment(basis, d) for d in random_dirs])
    p_value = permutation_pvalue(null >= observed)
    return {
        "alignment": float(observed), "angle_deg": float(np.degrees(np.arccos(np.clip(observed, -1.0, 1.0)))),
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)), "p_value": float(p_value),
        "n_rotation_draws": int(n_draws),
    }


def pool_sessions(values_by_unit: dict, n_null_draws_default: int) -> dict:
    """Cross-session/participant pooling: this project's shared two-sided
    paired sign-flip estimator (src.statistics.paired_sign_flip_test) of the
    per-unit normalised displacement against a null of zero, with its CI,
    the between-unit standard deviation, and the minimum detectable
    difference at 80% power (src.statistics.minimum_detectable_paired_difference).
    """
    units = sorted(values_by_unit)
    values = np.array([values_by_unit[u] for u in units], dtype=float)
    finite = np.isfinite(values)
    values, units = values[finite], [u for u, keep in zip(units, finite) if keep]
    if len(values) < 2:
        return {"status": "not_estimable", "reason": f"only {len(values)} units with a finite value",
                "n": int(len(values)), "units": units}
    rng = np.random.default_rng(stable_seed("stimulation_latent_response_map_pool_" + "_".join(units)))
    test = paired_sign_flip_test(values, np.zeros_like(values), alternative="two-sided", rng=rng)
    test.pop("null", None)
    mdd = minimum_detectable_paired_difference(values)
    return {
        "status": "complete", "n": int(len(values)), "units": units,
        "mean_diff": test["mean_diff"], "ci_lower": test["ci_lower"], "ci_upper": test["ci_upper"],
        "p_value": test["p_value"], "sd": mdd.get("sd"), "mdd_80pct_power": mdd.get("mdd"),
        "significant": bool(test["ci_lower"] > 0 or test["ci_upper"] < 0),
    }


def zero_drop_ledger() -> dict:
    return {"seen": [], "loaded": [], "refused": [], "computed": [], "loaded_but_not_computed": []}


def ledger_reconcile(ledger: dict) -> dict:
    seen, loaded = set(ledger["seen"]), set(ledger["loaded"])
    refused = {r["id"] for r in ledger["refused"]}
    computed = set(ledger["computed"])
    stalled = {r["id"] for r in ledger["loaded_but_not_computed"]}
    return {
        "n_seen": len(seen), "n_loaded": len(loaded), "n_refused": len(refused),
        "n_computed": len(computed), "n_loaded_but_not_computed": len(stalled),
        "seen_eq_loaded_plus_refused": seen == (loaded | refused) and len(loaded & refused) == 0,
        "loaded_eq_computed_plus_stalled": loaded == (computed | stalled) and len(computed & stalled) == 0,
    }


# ── Reproduction gate ────────────────────────────────────────────────────────────

def _float_close(a: float, b: float, rel_tol: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


VOLATILE_FIELD_NAMES = {"runtime_seconds"}  # wall-clock bookkeeping, never a claim this gate reproduces


def deep_diff(a, b, path: str = "$") -> list[str]:
    """Every mismatch between two JSON-safe structures, floats compared at
    tolerance 1e-6 (relative, with a small absolute floor for near-zero
    values), everything else by exact equality. Fields in
    VOLATILE_FIELD_NAMES (wall-clock timing, not a scientific claim) are
    skipped -- every other field is compared at full tolerance, unweakened."""
    diffs = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in VOLATILE_FIELD_NAMES:
                continue
            if key not in a:
                diffs.append(f"{path}.{key}: missing in reproduced output")
            elif key not in b:
                diffs.append(f"{path}.{key}: missing in delivered artifact")
            else:
                diffs.extend(deep_diff(a[key], b[key], f"{path}.{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f"{path}: length {len(a)} != {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs.extend(deep_diff(x, y, f"{path}[{i}]"))
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if not _float_close(float(a), float(b)):
            diffs.append(f"{path}: {a!r} != {b!r}")
    elif a != b:
        diffs.append(f"{path}: {a!r} != {b!r}")
    return diffs


def reproduce_arm(module_name: str, artifact_name: str, extra_inputs: list[str] | None = None) -> dict:
    """Re-run a delivered pipeline's own main() with its module-level
    RESULTS directory redirected to an isolated temp directory (so this
    check can never modify a delivered results/*.json), then diff the
    freshly-produced artifact against the real delivered one at tolerance
    1e-6. `extra_inputs` are delivered results files that pipeline reads as
    input (not output) and must see a real copy of, in the temp directory.
    """
    delivered_path = RESULTS / f"{artifact_name}.json"
    if not delivered_path.exists():
        return {"status": "void_reproduction_gate_did_not_reproduce",
                "reason": f"no delivered artifact at {delivered_path}"}
    delivered = json.loads(delivered_path.read_text())

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"repro_{module_name}_"))
    try:
        for name in (extra_inputs or []):
            src = RESULTS / name
            if src.exists():
                shutil.copy(src, tmp_dir / name)
        mod = importlib.import_module(module_name)
        importlib.reload(mod)  # a fresh module object, so a prior monkeypatch of RESULTS never leaks in
        original_results = mod.RESULTS
        original_argv = sys.argv
        mod.RESULTS = tmp_dir
        # run_ram_stimulation_drift.main() reads its own CLI flags via
        # argparse.parse_args() with no explicit list, which defaults to
        # this PROCESS's sys.argv -- this leg's own flags (e.g.
        # --skip-reproduction-gate) are not valid options for that parser.
        # An empty argv makes it fall back to its own delivered defaults
        # (its docstring's "--openloop-max-subjects 38
        # --closedloop-max-subjects 35"), which is exactly "re-run on its
        # delivered sessions". Harmless for the other three modules, which
        # never call parse_args() at all.
        sys.argv = [str(getattr(mod, "__file__", module_name))]
        try:
            mod.main()
        finally:
            mod.RESULTS = original_results
            sys.argv = original_argv
        produced_path = tmp_dir / f"{artifact_name}.json"
        if not produced_path.exists():
            return {"status": "void_reproduction_gate_did_not_reproduce",
                    "reason": f"{module_name}.main() did not write {artifact_name}.json"}
        produced = json.loads(produced_path.read_text())
        diffs = deep_diff(produced, delivered)
        if diffs:
            return {"status": "void_reproduction_gate_did_not_reproduce",
                    "reason": f"{len(diffs)} field(s) exceeded tolerance 1e-6",
                    "n_mismatches": len(diffs), "examples": diffs[:20]}
        return {"status": "reproduced", "tolerance": "1e-6 (math.isclose rel_tol, abs_tol=1e-9 floor)"}
    except Exception as exc:  # noqa: BLE001 -- a reproduction failure is a result, not a crash
        return {"status": "void_reproduction_gate_did_not_reproduce",
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def reproduction_gate_source_drift(repo_root: Path) -> dict:
    """Whether the estimator source files the reproduction gate depends on
    are themselves unchanged since the commit currently checked out --
    context for reading a void_reproduction_gate_did_not_reproduce status:
    a void result caused by the analysis source having moved on since a
    delivered artifact was generated is a different finding from a void
    result caused by genuine run-to-run non-determinism, and this
    distinguishes which files (if any) are implicated."""
    watched = ["src/dynamics.py", "src/geometry.py", "src/control.py", "src/drift_dynamics.py",
              "src/preprocessing.py", "src/statistics.py"]
    modified = []
    for rel in watched:
        try:
            out = subprocess.run(["git", "-C", str(repo_root), "diff", "--stat", "--", rel],
                                 capture_output=True, text=True, timeout=30)
            if out.returncode == 0 and out.stdout.strip():
                modified.append(rel)
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {"watched_files": watched, "modified_relative_to_head": modified,
            "head_commit": git_commit(repo_root)}


def reproduction_gate_mismatch_pattern(gate_result: dict) -> dict:
    """Classify a void arm's mismatch examples into the two patterns this
    analysis's own investigation found: fields present in the fresh run and
    absent from the delivered artifact (a source change that added outputs,
    e.g. dominant-mode classification fields), versus small relative
    numeric differences on fields that exist in both (consistent with
    floating-point non-determinism in a multi-threaded fit rather than a
    methodology change)."""
    examples = gate_result.get("examples") or []
    added_fields = [e for e in examples if "missing in delivered artifact" in e]
    changed_values = [e for e in examples if "missing in delivered artifact" not in e
                      and "missing in reproduced output" not in e and "length" not in e]
    return {
        "n_mismatches_total": gate_result.get("n_mismatches"),
        "n_fields_present_only_in_fresh_run_among_examples": len(added_fields),
        "n_changed_values_among_examples": len(changed_values),
        "example_added_fields": added_fields[:5], "example_changed_values": changed_values[:5],
    }


def reproduction_gate_investigation(reproduction_gate: dict, repo_root: Path) -> dict:
    """This leg's own investigation into WHY three of the four delivered
    pipelines failed to reproduce -- required reading before treating either
    the delivered numbers or the freshly reproduced ones as ground truth.
    Never edits the implicated source files or the delivered artifacts;
    only documents what was found.
    """
    source_drift = reproduction_gate_source_drift(repo_root)
    findings = {"source_files_modified_since_last_commit": source_drift}
    void_arms = {name: res for name, res in reproduction_gate.items()
                if res.get("status") == "void_reproduction_gate_did_not_reproduce" and res.get("examples")}
    if not void_arms:
        findings["note"] = "every reproduction check either reproduced or was skipped; no drift to investigate"
        return findings

    per_arm = {name: reproduction_gate_mismatch_pattern(res) for name, res in void_arms.items()}
    findings["per_arm_mismatch_pattern"] = per_arm

    dynamics_or_geometry_touched = ("src/dynamics.py" in source_drift["modified_relative_to_head"]
                                    or "src/geometry.py" in source_drift["modified_relative_to_head"])
    structural_arms = [name for name, pat in per_arm.items()
                       if pat["n_fields_present_only_in_fresh_run_among_examples"] > 0]
    numeric_only_arms = [name for name, pat in per_arm.items()
                         if pat["n_fields_present_only_in_fresh_run_among_examples"] == 0
                         and pat["n_changed_values_among_examples"] > 0]

    if structural_arms and dynamics_or_geometry_touched:
        findings["structural_mismatch_arms"] = {
            "arms": structural_arms,
            "finding": ("These arms fit a dynamics model whose output fields (dominant-mode classification, "
                       "rho, theta) do not exist in the delivered artifact at all -- the fresh run's fields "
                       "are a strict superset. src/dynamics.py and/or src/geometry.py are modified in this "
                       "working tree relative to the currently checked-out commit (see "
                       "source_files_modified_since_last_commit), so the delivered artifact was generated "
                       "from an earlier state of that source than what the reproduction gate runs today. "
                       "One example changed value in this category is a null-model R-squared that was "
                       "reported near 1.0 in the delivered artifact and lands near zero (or negative) in the "
                       "fresh run -- a near-zero or negative R-squared is the statistically expected behaviour "
                       "of a held-out, honestly-evaluated null on shuffled data, and a null-model fit that "
                       "scores nearly as well as the real fit is the signature of a null being evaluated "
                       "in-sample rather than held out. This does not establish which side is correct in "
                       "general, but it does mean the delivered artifact's own reported number for this "
                       "specific quantity is not a safe reference to force a match against, and no such "
                       "match was forced (tolerance was not loosened; these arms are reported void)."),
        }
    if numeric_only_arms:
        findings["numeric_only_mismatch_arms"] = {
            "arms": numeric_only_arms,
            "finding": ("These arms show small relative differences on fields that exist in both the fresh "
                       "and delivered outputs, with no fields added or removed -- a different signature from "
                       "the structural case above, more consistent with floating-point run-to-run "
                       "non-determinism in a multi-threaded fit (the delivered pipeline's own default runs a "
                       "multi-worker spectral transform) than with a methodology change. This is reported as "
                       "the best-evidenced hypothesis available without a controlled single-threaded rerun, "
                       "not a confirmed diagnosis."),
        }
    return findings


# ── Arm 1 & 2: ds005489 (open-loop) and ds005557 (closed-loop) ─────────────────

def _ram_lightweight_frame_pc1(features: np.ndarray, mask: np.ndarray, n_pc: int) -> np.ndarray | None:
    """Refit the delivered arm's own z-scored-PCA frame (RAM.fit_group_drift's
    frame construction, exactly) on `mask` trials and project every trial's
    leading (PC1) trajectory -- the delivered arm's own session-specific
    unsupervised leading component, kept as this leg's frame for this arm
    (RAM never fits a multi-dimensional dynamics model). Skips
    fit_group_drift's own confined-diffusion model fit, which this leg's
    normalisation does not use, so a within-session null of >=1000 draws is
    tractable. Returns None with too few plant trials to fit a frame."""
    if mask.sum() < 3:
        return None
    n_trials, n_bins, n_feat = features.shape
    train = features[mask].reshape(-1, n_feat)
    mu, sd = train.mean(axis=0), train.std(axis=0) + 1e-8
    z = (features - mu) / sd
    z_train = z[mask].reshape(-1, n_feat)
    _, components, _ = pca_decompose(z_train, n_pc)
    proj = ((z.reshape(-1, n_feat) - z_train.mean(axis=0)) @ components).reshape(n_trials, n_bins, components.shape[1])
    return proj[:, :, 0]


def _ram_session_estimator(features: np.ndarray, unstim_mask: np.ndarray, stim_mask: np.ndarray,
                            n_pc: int) -> dict:
    pc1 = _ram_lightweight_frame_pc1(features, unstim_mask, n_pc)
    if pc1 is None:
        return {"status": "not_estimable", "reason": "fewer than 3 unstimulated trials"}
    scores = pc1[:, :, None]
    centroid = np.array([pc1[unstim_mask].mean()])
    state = per_trial_state_norm(scores, centroid)
    return displacement_and_normalisation(state, unstim_mask, stim_mask)


def ram_openloop_session(session: dict, n_pc: int, n_null_draws: int) -> dict:
    features, stim, recalled = session["features"], session["stim"], session["recalled"]
    stim_list = session["stim_list"]
    n = len(stim)
    unstim_mask = stim == 0
    stim_mask = (stim == 1) & (stim_list == 1)
    est = _ram_session_estimator(features, unstim_mask, stim_mask, n_pc)
    if est["status"] != "complete":
        return {"status": est["status"], "reason": est.get("reason"), "session": session["session"]}

    rng = np.random.default_rng(stable_seed(f"ram_openloop_null_{session['session']}"))

    def _fake(u_mask, s_mask):
        r = _ram_session_estimator(features, u_mask, s_mask, n_pc)
        return r.get("normalized_displacement") if r["status"] == "complete" else None

    null = within_unit_permutation_null(_fake, int(unstim_mask.sum()), int(stim_mask.sum()), n, n_null_draws, rng)
    obs = est["normalized_displacement"]
    within_arm_null = None
    if null is not None and obs is not None:
        finite_null = null[np.isfinite(null)]
        if len(finite_null) >= 10:
            within_arm_null = {
                "n_draws": int(len(finite_null)), "null_mean": float(finite_null.mean()),
                "null_sd": float(finite_null.std(ddof=1)),
                "p_value": float(permutation_pvalue(finite_null >= obs)),
            }

    behavior = None
    rec_stim = recalled[stim_mask]
    disp_stim_mask_idx = np.flatnonzero(stim_mask)
    if len(disp_stim_mask_idx) >= 6 and 0 < rec_stim.sum() < len(rec_stim):
        pc1 = _ram_lightweight_frame_pc1(features, unstim_mask, n_pc)
        centroid = np.array([pc1[unstim_mask].mean()])
        state = per_trial_state_norm(pc1[:, :, None], centroid)
        recalled_state = state[stim_mask][rec_stim == 1]
        forgotten_state = state[stim_mask][rec_stim == 0]
        behavior = {"status": "complete", "mean_state_recalled": float(recalled_state.mean()),
                    "mean_state_forgotten": float(forgotten_state.mean()),
                    "n_recalled": int(len(recalled_state)), "n_forgotten": int(len(forgotten_state))}
    else:
        behavior = {"status": "not_estimable", "reason": f"{int(rec_stim.sum())} recalled / "
                    f"{len(rec_stim) - int(rec_stim.sum())} forgotten stimulated trials"}

    return {"status": "complete", "session": session["session"], "subject": session["subject"],
            "displacement": est, "within_arm_null": within_arm_null, "behavior_by_recall": behavior}


def ram_closedloop_session(session: dict, n_pc: int, n_null_draws: int) -> dict:
    features, stim, recalled = session["features"], session["stim"], session["recalled"]
    stim_list = session["stim_list"]
    n = len(stim)
    unstim_mask = stim == 0  # the delivered arm's own plant mask, regardless of list

    # List-level: causal (list assignment is randomized).
    list_stim_mask = stim_list == 1
    list_est = _ram_session_estimator(features, unstim_mask, list_stim_mask, n_pc)
    list_null = None
    if list_est["status"] == "complete":
        rng = np.random.default_rng(stable_seed(f"ram_closedloop_list_null_{session['session']}"))

        def _fake_list(u_mask, s_mask):
            r = _ram_session_estimator(features, u_mask, s_mask, n_pc)
            return r.get("normalized_displacement") if r["status"] == "complete" else None

        draws = within_unit_permutation_null(_fake_list, int(unstim_mask.sum()), int(list_stim_mask.sum()),
                                              n, n_null_draws, rng)
        obs = list_est["normalized_displacement"]
        if draws is not None and obs is not None:
            finite = draws[np.isfinite(draws)]
            if len(finite) >= 10:
                list_null = {"n_draws": int(len(finite)), "null_mean": float(finite.mean()),
                             "null_sd": float(finite.std(ddof=1)), "p_value": float(permutation_pvalue(finite >= obs))}

    # Item-level: descriptive only, never causal (propensity-selected on the measured state).
    triggered_mask = (stim_list == 1) & (stim == 1)
    nontriggered_mask = (stim_list == 1) & (stim == 0)
    item_est = {"status": "not_estimable",
                "reason": f"{int(triggered_mask.sum())} triggered / {int(nontriggered_mask.sum())} nontriggered"}
    item_null = None
    if triggered_mask.sum() >= 1 and nontriggered_mask.sum() >= 3:
        pc1 = _ram_lightweight_frame_pc1(features, unstim_mask, n_pc)
        if pc1 is not None:
            centroid = np.array([pc1[unstim_mask].mean()])
            state = per_trial_state_norm(pc1[:, :, None], centroid)
            spontaneous_sd = float(state[unstim_mask].std(ddof=1))
            raw = float(state[triggered_mask].mean() - state[nontriggered_mask].mean())
            item_est = {
                "status": "complete", "causal": False, "descriptive_only": True,
                "why_noncausal": ("Item-level stimulation is triggered only when the classifier's own "
                                  "encoding-window output falls below the running median -- assignment is "
                                  "propensity-selected on the very state measured here, not randomized."),
                "n_triggered": int(triggered_mask.sum()), "n_nontriggered": int(nontriggered_mask.sum()),
                "raw_displacement": raw,
                "normalized_displacement": raw / spontaneous_sd if spontaneous_sd > 0 else None,
                "spontaneous_sd_raw": spontaneous_sd,
            }
            rng = np.random.default_rng(stable_seed(f"ram_closedloop_item_null_{session['session']}"))
            list_pool = np.flatnonzero(stim_list == 1)

            def _fake_item(fake_triggered_local, fake_nontriggered_local):
                sub = state[list_pool]
                r_disp = float(sub[fake_triggered_local].mean() - sub[fake_nontriggered_local].mean())
                sd = float(state[unstim_mask].std(ddof=1))
                return r_disp / sd if sd > 0 else None

            n_trig, n_pool = int(triggered_mask.sum()), len(list_pool)
            draws = np.full(n_null_draws, np.nan)
            if n_trig >= 1 and n_pool - n_trig >= 3:
                for d in range(n_null_draws):
                    perm = rng.permutation(n_pool)
                    trig_local = np.zeros(n_pool, dtype=bool)
                    trig_local[perm[:n_trig]] = True
                    nontrig_local = ~trig_local
                    value = _fake_item(trig_local, nontrig_local)
                    draws[d] = value if value is not None else np.nan
                finite = draws[np.isfinite(draws)]
                if len(finite) >= 10 and item_est["normalized_displacement"] is not None:
                    item_null = {"n_draws": int(len(finite)), "null_mean": float(finite.mean()),
                                 "null_sd": float(finite.std(ddof=1)),
                                 "p_value": float(permutation_pvalue(finite >= item_est["normalized_displacement"]))}

    behavior = None
    if list_stim_mask.sum() >= 6:
        pc1 = _ram_lightweight_frame_pc1(features, unstim_mask, n_pc)
        if pc1 is not None:
            centroid = np.array([pc1[unstim_mask].mean()])
            state = per_trial_state_norm(pc1[:, :, None], centroid)
            rec = recalled[list_stim_mask]
            if 0 < rec.sum() < len(rec):
                s = state[list_stim_mask]
                behavior = {"status": "complete", "mean_state_recalled": float(s[rec == 1].mean()),
                            "mean_state_forgotten": float(s[rec == 0].mean()),
                            "n_recalled": int(rec.sum()), "n_forgotten": int(len(rec) - rec.sum())}
    if behavior is None:
        behavior = {"status": "not_estimable", "reason": "insufficient recalled/forgotten list-level stim trials"}

    return {"status": "complete", "session": session["session"], "subject": session["subject"],
            "list_level": {"displacement": list_est, "within_arm_null": list_null},
            "item_level": {"displacement": item_est, "within_arm_null": item_null},
            "behavior_by_recall_list_level": behavior}


def ram_arm(dataset_key: str, max_units: int, derive_stim: bool, per_session_fn, label: str,
            n_pc: int, n_null_draws: int, n_jobs: int) -> dict:
    from run_ram_stimulation_drift import _find_stim_sessions, build_session_trajectory, configured_data_path

    data_root = configured_data_path(dataset_key)
    sessions = _find_stim_sessions(data_root)
    ledger = zero_drop_ledger()
    per_session_records = {}
    done_subjects = set()
    for ieeg_json in sessions:
        subj = ieeg_json.parts[-4]
        session_id = str(ieeg_json.relative_to(data_root))
        if len(done_subjects) >= max_units and subj not in done_subjects:
            continue
        ledger["seen"].append(session_id)

        def _fit(ieeg_json=ieeg_json, subj=subj, session_id=session_id):
            try:
                session = build_session_trajectory(ieeg_json, data_root, derive_stim, n_jobs=n_jobs)
            except Exception as exc:  # noqa: BLE001
                return {"loaded": False, "reason": f"processing error: {exc}", "session_id": session_id, "subject": subj}
            if session is None:
                return {"loaded": False, "reason": "no stimulation channel (out of scope)", "session_id": session_id, "subject": subj}
            if session["status"] != "complete":
                return {"loaded": False, "reason": session["reason"], "session_id": session_id, "subject": subj}
            result = per_session_fn(session, n_pc, n_null_draws)
            computed = result.get("status") == "complete"
            return {"loaded": True, "computed": computed, "reason": None if computed else result.get("reason"),
                    "session_id": session_id, "subject": subj, "result": result}

        outcome = run_checkpointed(f"{label}__{session_id}", _fit)
        if not outcome["loaded"]:
            ledger["refused"].append({"id": session_id, "reason": outcome["reason"]})
            continue
        ledger["loaded"].append(session_id)
        done_subjects.add(subj)
        if outcome["computed"]:
            ledger["computed"].append(session_id)
            per_session_records[session_id] = outcome["result"]
        else:
            ledger["loaded_but_not_computed"].append({"id": session_id, "reason": outcome["reason"]})

    return {"label": label, "n_units_requested": max_units, "ledger": ledger,
            "ledger_reconciliation": ledger_reconcile(ledger), "per_session": per_session_records}


# ── Arm 4: macaque_pfc_microstimulation (macaque delay-period microstimulation) ───────────────

def _macaque_pfc_microstimulation_lightweight_frame(pooled_trials: np.ndarray, unstim_mask: np.ndarray, n_pc: int):
    """pooled_trials: (N, n_bins, C). Refits src.geometry.pca_decompose on the
    unstim subset (the same primitive, same n_pc, the delivered macaque_pfc_microstimulation
    pipeline's own frame construction uses) and projects every trial."""
    if unstim_mask.sum() < 5:
        return None, None, None
    N, n_bins, C = pooled_trials.shape
    train = pooled_trials[unstim_mask].reshape(-1, C)
    _, components, _ = pca_decompose(train, n_pc)
    mean = train.mean(axis=0)
    proj = ((pooled_trials.reshape(-1, C) - mean) @ components).reshape(N, n_bins, components.shape[1])
    return proj, components, mean


def macaque_pfc_microstimulation_arm(max_sessions: int, n_pc_default: int, n_null_draws: int, n_rotation_null: int) -> dict:
    from run_macaque_pfc_microstimulation_pipeline import BIN_S, N_PC, SESSIONS, crop_trial, load_macaque_pfc_microstimulation_session

    delivered_path = RESULTS / "causal_macaque_pfc_microstimulation.json"
    delivered = json.loads(delivered_path.read_text()) if delivered_path.exists() else {"per_session": {}}
    n_pc = N_PC if N_PC else n_pc_default

    ledger = zero_drop_ledger()
    per_session_records = {}
    for prefix in SESSIONS[:max_sessions]:
        ledger["seen"].append(prefix)

        def _fit(prefix=prefix):
            corr = load_macaque_pfc_microstimulation_session(prefix, correct=True)
            if corr is None or corr["control_idx"] is None:
                return {"loaded": False, "reason": "session not loadable or no control condition"}
            err = load_macaque_pfc_microstimulation_session(prefix, correct=False)
            control_idx, channel_ids = corr["control_idx"], corr["channel_ids"]

            def epochs_for(source, cond):
                return [crop_trial(tr["spikerate"]) for tr in source["trials"] if tr["stim_cond"] == cond]

            ctrl_correct = [e for e in epochs_for(corr, control_idx) if e is not None]
            if len(ctrl_correct) < 10:
                return {"loaded": False, "reason": f"only {len(ctrl_correct)} control-correct trials (< 10)"}
            Z_ctrl = np.stack(ctrl_correct, axis=0)

            rng = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_map_{prefix}"))
            # Z_ctrl is (N, N_BINS, C) -- this arm's own native trial layout
            # (crop_trial returns (time, channel); see run_macaque_pfc_microstimulation_pipeline.py).
            # fit_retention_dynamics requires (N, C, T); transpose here rather
            # than at every other call site, which already treats axis 1 as
            # bins and axis 2 as channels consistently with the delivered
            # pipeline's own convention.
            dyn = fit_retention_dynamics(Z_ctrl.transpose(0, 2, 1), srate=1.0 / BIN_S, k=n_pc, rng=rng)
            dominant = dominant_eigenmode(dyn["A"])
            subspace_m2 = invariant_subspace_basis(dyn["A"], min(2, dyn["A"].shape[0]))

            all_ctrl = ctrl_correct + [e for e in epochs_for(err, control_idx) if e is not None] if err is not None else ctrl_correct
            Z_all_ctrl = np.stack(all_ctrl, axis=0)
            unstim_mask_full = np.ones(len(all_ctrl), dtype=bool)
            proj_ctrl, components, mean = _macaque_pfc_microstimulation_lightweight_frame(Z_all_ctrl, unstim_mask_full, n_pc)
            if proj_ctrl is None:
                return {"loaded": False, "reason": "fewer than 5 control trials for the spontaneous-variability frame"}
            centroid = proj_ctrl.mean(axis=(0, 1))
            state_ctrl = per_trial_state_norm(proj_ctrl, centroid)
            spontaneous_sd = float(state_ctrl.std(ddof=1))
            baseline_dispersion = float(state_ctrl.mean())
            ctrl_vec_mean = proj_ctrl.mean(axis=1).mean(axis=0)  # (k,)

            delivered_session = delivered.get("per_session", {}).get(prefix, {})
            delivered_cond = delivered_session.get("cond_features", {})

            # Task-relevant coding subspace: class centroids of the remembered
            # target angle, fit on the same control-correct trials.
            angle_by_trial = np.array([tr["angle_idx"] for tr in corr["trials"] if tr["stim_cond"] == control_idx
                                        and crop_trial(tr["spikerate"]) is not None])
            content_subspace = None
            if len(np.unique(angle_by_trial)) >= 2 and len(angle_by_trial) == len(ctrl_correct):
                proj_ctrl_correct_only, _, _ = _macaque_pfc_microstimulation_lightweight_frame(
                    np.stack(ctrl_correct, axis=0), np.ones(len(ctrl_correct), dtype=bool), n_pc)
                trial_vecs = proj_ctrl_correct_only.mean(axis=1)  # (n, k)
                classes = np.unique(angle_by_trial)
                class_means = np.stack([trial_vecs[angle_by_trial == c].mean(axis=0) for c in classes])
                m = int(min(len(classes) - 1, class_means.shape[1]))
                if m >= 1:
                    _, class_basis, _ = pca_decompose(class_means - class_means.mean(axis=0), m)
                    content_subspace = {"n_classes": int(len(classes)), "subspace_dim": m, "class_basis": class_basis}

            conditions = {}
            for c in range(len(corr["stim_channels"])):
                if c == control_idx:
                    continue
                stim_epochs = [e for e in epochs_for(corr, c) if e is not None]
                if err is not None:
                    stim_epochs += [e for e in epochs_for(err, c) if e is not None]
                if len(stim_epochs) < 5:
                    continue
                Z_stim = np.stack(stim_epochs, axis=0)
                proj_stim = ((Z_stim.reshape(-1, Z_stim.shape[-1]) - mean) @ components).reshape(
                    Z_stim.shape[0], Z_stim.shape[1], components.shape[1])
                state_stim = per_trial_state_norm(proj_stim, centroid)
                raw = float(state_stim.mean() - baseline_dispersion)
                normalized = raw / spontaneous_sd if spontaneous_sd > 0 else None
                stim_vec_mean = proj_stim.mean(axis=1).mean(axis=0)
                direction = stim_vec_mean - ctrl_vec_mean
                direction_norm = float(np.linalg.norm(direction))
                direction_unit = direction / direction_norm if direction_norm > 1e-12 else None

                cond_rec = {
                    "n_stim_trials": int(len(stim_epochs)), "raw_displacement": raw,
                    "normalized_displacement": normalized, "spontaneous_sd_raw": spontaneous_sd,
                    "displacement_direction_norm": direction_norm,
                    "displacement_direction_unit": (
                        direction_unit.tolist() if direction_unit is not None else None
                    ),
                    "delivered_stimulation_input_alignment": {
                        k_: delivered_cond.get(str(c), {}).get(k_)
                        for k_ in ("alignment_to_vstar", "gramian_trace", "stable_alignment",
                                   "random_alignment", "within_frac", "outside_frac")
                    },
                }
                if direction_unit is not None:
                    cond_rec["displacement_vs_vstar"] = rotation_null_alignment(
                        dyn["v_star"].reshape(-1, 1), direction_unit, n_rotation_null,
                        np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_vstar_null_{prefix}_{c}")))
                    cond_rec["displacement_vs_dynamic_subspace_m2"] = rotation_null_alignment(
                        subspace_m2.basis, direction_unit, n_rotation_null,
                        np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_m2_null_{prefix}_{c}")))
                    if content_subspace is not None:
                        cond_rec["displacement_vs_content_subspace"] = rotation_null_alignment(
                            content_subspace["class_basis"], direction_unit, n_rotation_null,
                            np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_content_null_{prefix}_{c}")))
                        cond_rec["content_subspace_n_classes"] = content_subspace["n_classes"]
                        cond_rec["content_subspace_dim"] = content_subspace["subspace_dim"]
                    else:
                        cond_rec["displacement_vs_content_subspace"] = {
                            "status": "not_applicable",
                            "reason": "fewer than 2 target-angle classes with usable control trials this session"}

                # Within-session null: permute stim_cond labels within the
                # (all-control ∪ this condition's) trial pool.
                pool = np.concatenate([Z_all_ctrl, Z_stim], axis=0)
                n_ctrl_pool, n_stim_pool = len(Z_all_ctrl), len(Z_stim)
                rng_null = np.random.default_rng(stable_seed(f"macaque_pfc_microstimulation_disp_null_{prefix}_{c}"))

                def _fake(u_mask, s_mask, pool=pool):
                    proj, comp, mn = _macaque_pfc_microstimulation_lightweight_frame(pool, u_mask, n_pc)
                    if proj is None:
                        return None
                    cen = proj[u_mask].mean(axis=(0, 1))
                    st = per_trial_state_norm(proj, cen)
                    return displacement_and_normalisation(st, u_mask, s_mask).get("normalized_displacement")

                draws = within_unit_permutation_null(_fake, n_ctrl_pool, n_stim_pool, n_ctrl_pool + n_stim_pool,
                                                      n_null_draws, rng_null)
                if draws is not None and normalized is not None:
                    finite = draws[np.isfinite(draws)]
                    if len(finite) >= 10:
                        cond_rec["within_arm_null"] = {
                            "n_draws": int(len(finite)), "null_mean": float(finite.mean()),
                            "null_sd": float(finite.std(ddof=1)),
                            "p_value": float(permutation_pvalue(finite >= normalized))}
                conditions[str(c)] = cond_rec

            return {"loaded": True, "computed": len(conditions) > 0,
                    "reason": None if conditions else "no stimulation condition had >=5 usable trials",
                    "dynamics": {"rho": dyn["rho"], "theta": dyn["theta"], "classification": dyn["classification"],
                                 "dominant_eigenmode_classification_direct_call": dominant.classification,
                                 "r2_cv": dyn["r2_cv"], "r2_null": dyn["r2_null"], "identifiable": dyn["identifiable"]},
                    "baseline_dispersion_raw": baseline_dispersion, "spontaneous_sd_raw": spontaneous_sd,
                    "n_control_trials_for_frame": len(all_ctrl), "conditions": conditions}

        outcome = run_checkpointed(f"macaque_pfc_microstimulation__{prefix}", _fit)
        if not outcome["loaded"]:
            ledger["refused"].append({"id": prefix, "reason": outcome["reason"]})
            continue
        ledger["loaded"].append(prefix)
        if outcome["computed"]:
            ledger["computed"].append(prefix)
            per_session_records[prefix] = outcome
        else:
            ledger["loaded_but_not_computed"].append({"id": prefix, "reason": outcome["reason"]})

    return {"ledger": ledger, "ledger_reconciliation": ledger_reconcile(ledger), "per_session": per_session_records}


# ── Arm 3: haslacher_clam_tacs (non-invasive phase-tuned tACS) ─────────────────

def haslacher_arm(max_active: int, max_control: int, n_pc_method: str, n_null_draws: int,
                   n_rotation_null: int) -> dict:
    from run_haslacher_stimulation_geometry import (
        ACTIVE_SUBJECTS, CONTROL_SUBJECTS, GRAMIAN_HORIZON, N_RANDOM_DIRS, PHASE_CONDITIONS,
        _preprocess, _retention_trials, _stimulation_channel_weight,
    )
    from geometry import select_latent_dim

    phase_omega_path = RESULTS / "haslacher_phase_omega.json"
    behavior_by_subject = {}
    if phase_omega_path.exists():
        behavior_json = json.loads(phase_omega_path.read_text())
        for group_name in ("active", "control"):
            for subj, rec in behavior_json.get(group_name, {}).get("per_subject", {}).items():
                depth = rec.get("modulation", {}).get("depth")
                if depth is not None:
                    behavior_by_subject[subj] = depth

    delivered_path = RESULTS / "haslacher_stimulation_geometry.json"
    delivered = json.loads(delivered_path.read_text()) if delivered_path.exists() else {}

    ledger = zero_drop_ledger()
    per_subject_records = {}
    subjects = [(s, "active") for s in ACTIVE_SUBJECTS[:max_active]] + [(s, "control") for s in CONTROL_SUBJECTS[:max_control]]
    for subject, group in subjects:
        ledger["seen"].append(subject)

        def _fit(subject=subject, group=group):
            try:
                no_stim, stim, n_sass = _preprocess(subject, group)
            except (FileNotFoundError, OSError) as e:
                return {"loaded": False, "reason": str(e)}
            baseline_trials = _retention_trials(no_stim)[0]
            condition_trials = _retention_trials(stim, codes=list(PHASE_CONDITIONS))
            n_stim_total = sum(len(v) for v in condition_trials.values())
            if baseline_trials.shape[0] < 3 or n_stim_total < 3:
                return {"loaded": False, "reason": f"{baseline_trials.shape[0]} baseline / {n_stim_total} stim trials"}

            rng = np.random.default_rng(stable_seed(f"haslacher_map_{subject}"))
            k_info = select_latent_dim(baseline_trials, method=n_pc_method, rng=rng)
            k = k_info["k"]
            srate = float(no_stim.info["sfreq"])
            dyn = fit_retention_dynamics(baseline_trials, srate, k, rng)
            dominant = dominant_eigenmode(dyn["A"])
            subspace_m2 = invariant_subspace_basis(dyn["A"], min(2, dyn["A"].shape[0]))

            baseline_pooled = baseline_trials.transpose(0, 2, 1).reshape(-1, baseline_trials.shape[1])
            scores_baseline = ((baseline_pooled - dyn["mean"]) @ dyn["components"]).reshape(
                baseline_trials.shape[0], baseline_trials.shape[2], k)
            centroid = scores_baseline.mean(axis=(0, 1))
            state_baseline = per_trial_state_norm(scores_baseline, centroid)
            spontaneous_sd = float(state_baseline.std(ddof=1))
            baseline_dispersion = float(state_baseline.mean())
            baseline_vec_mean = scores_baseline.mean(axis=1).mean(axis=0)

            stim_pooled_trials = np.concatenate([v for v in condition_trials.values() if len(v)], axis=0)
            stim_flat = stim_pooled_trials.transpose(0, 2, 1).reshape(-1, stim_pooled_trials.shape[1])
            scores_stim = ((stim_flat - dyn["mean"]) @ dyn["components"]).reshape(
                stim_pooled_trials.shape[0], stim_pooled_trials.shape[2], k)
            state_stim = per_trial_state_norm(scores_stim, centroid)
            raw = float(state_stim.mean() - baseline_dispersion)
            normalized = raw / spontaneous_sd if spontaneous_sd > 0 else None
            stim_vec_mean = scores_stim.mean(axis=1).mean(axis=0)
            direction = stim_vec_mean - baseline_vec_mean
            direction_norm = float(np.linalg.norm(direction))
            direction_unit = direction / direction_norm if direction_norm > 1e-12 else None

            record = {
                "loaded": True, "computed": True, "group": group,
                "dynamics": {"rho": dyn["rho"], "classification": dyn["classification"],
                             "dominant_eigenmode_classification_direct_call": dominant.classification,
                             "r2_cv": dyn["r2_cv"], "r2_null": dyn["r2_null"], "identifiable": dyn["identifiable"]},
                "n_baseline_trials": int(baseline_trials.shape[0]), "n_stim_trials": int(n_stim_total),
                "baseline_dispersion_raw": baseline_dispersion, "spontaneous_sd_raw": spontaneous_sd,
                "raw_displacement": raw, "normalized_displacement": normalized,
                "displacement_direction_norm": direction_norm,
                "displacement_direction_unit": (
                    direction_unit.tolist() if direction_unit is not None else None
                ),
                "content_subspace_alignment": {
                    "status": "not_applicable",
                    "reason": "no supervised memory-content label in the delivered pipeline for this arm "
                              "(phase condition is a stimulation-timing label, not a memory content label)",
                },
            }
            if direction_unit is not None:
                record["displacement_vs_vstar"] = rotation_null_alignment(
                    dyn["v_star"].reshape(-1, 1), direction_unit, n_rotation_null,
                    np.random.default_rng(stable_seed(f"haslacher_vstar_null_{subject}")))
                record["content_subspace_alignment"]["unsupervised_alternative"] = "alignment to v_star (displacement_vs_vstar)"
                record["displacement_vs_dynamic_subspace_m2"] = rotation_null_alignment(
                    subspace_m2.basis, direction_unit, n_rotation_null,
                    np.random.default_rng(stable_seed(f"haslacher_m2_null_{subject}")))

            chan_weight = _stimulation_channel_weight(group, no_stim.ch_names)
            if chan_weight is not None:
                align_rng = np.random.default_rng(stable_seed(f"haslacher_map_stimalign_{subject}"))
                stim_input = stimulation_input_alignment(
                    dyn["A"], dyn["components"], chan_weight["weight"], dyn["v_star"], dyn["v_stable"], align_rng,
                    gramian_horizon=GRAMIAN_HORIZON, n_random_dirs=N_RANDOM_DIRS)
                record["stimulation_input_alignment_freshly_computed"] = stim_input
            record["stimulation_input_alignment_delivered"] = delivered.get(group, {}).get(
                "per_subject", {}).get(subject, {}).get("stimulation_input_alignment")
            record["within_manifold_outside_manifold_fraction"] = {
                "status": "not_computed_by_delivered_pipeline",
                "reason": "this arm's delivered pipeline never built a channel-space manifold projector "
                          "for the stimulation electrodes (unlike the macaque delay-period arm)",
            }

            rng_null = np.random.default_rng(stable_seed(f"haslacher_disp_null_{subject}"))
            pool = np.concatenate([baseline_trials, stim_pooled_trials], axis=0)
            n_base, n_stim_n = baseline_trials.shape[0], stim_pooled_trials.shape[0]

            def _fake(u_mask, s_mask, pool=pool):
                flat = pool.transpose(0, 2, 1).reshape(-1, pool.shape[1])
                trial_of_row = np.repeat(np.arange(len(pool)), pool.shape[2])
                u_rows = u_mask[trial_of_row]
                if u_mask.sum() < 3:
                    return None
                train = flat[u_rows]
                _, comp, _ = pca_decompose(train, k)
                mn = train.mean(axis=0)
                proj = ((flat - mn) @ comp).reshape(len(pool), pool.shape[2], comp.shape[1])
                cen = proj[u_mask].mean(axis=(0, 1))
                st = per_trial_state_norm(proj, cen)
                return displacement_and_normalisation(st, u_mask, s_mask).get("normalized_displacement")

            draws = within_unit_permutation_null(_fake, n_base, n_stim_n, n_base + n_stim_n, n_null_draws, rng_null)
            if draws is not None and normalized is not None:
                finite = draws[np.isfinite(draws)]
                if len(finite) >= 10:
                    record["within_arm_null"] = {
                        "n_draws": int(len(finite)), "null_mean": float(finite.mean()),
                        "null_sd": float(finite.std(ddof=1)), "p_value": float(permutation_pvalue(finite >= normalized))}

            record["behavioral_modulation_depth"] = behavior_by_subject.get(subject)
            return record

        outcome = run_checkpointed(f"haslacher__{subject}", _fit)
        if not outcome["loaded"]:
            ledger["refused"].append({"id": subject, "reason": outcome["reason"]})
            continue
        ledger["loaded"].append(subject)
        if outcome.get("computed"):
            ledger["computed"].append(subject)
            per_subject_records[subject] = outcome
        else:
            ledger["loaded_but_not_computed"].append({"id": subject, "reason": outcome.get("reason")})

    # Reused delivered brain-behavior link (already computed by the delivered
    # pipeline; cited here, not re-derived) plus this leg's own new
    # correlation between its normalised displacement and the same
    # behavioral-modulation-depth outcome the delivered pipeline defines.
    delivered_brain_behavior_link = delivered.get("brain_behavior_link")
    own_disp = [(sid, rec["normalized_displacement"]) for sid, rec in per_subject_records.items()
                if rec.get("normalized_displacement") is not None and sid in behavior_by_subject]
    behavioral_coupling = {"status": "not_estimable", "reason": f"only {len(own_disp)} participants with both values"}
    if len(own_disp) >= 3:
        x = np.array([v for _, v in own_disp])
        y = np.array([behavior_by_subject[sid] for sid, _ in own_disp])
        result = pearson_permutation_test(x, y)
        result["n"] = len(own_disp)
        behavioral_coupling = result

    return {"ledger": ledger, "ledger_reconciliation": ledger_reconcile(ledger), "per_subject": per_subject_records,
            "delivered_brain_behavior_link": delivered_brain_behavior_link,
            "own_normalized_displacement_vs_behavioral_modulation_depth": behavioral_coupling}


# ── Arm 3 disambiguation + Alagapan (distractor, reproduction only) ────────────

def disambiguate_arm3() -> dict:
    alagapan_path = RESULTS / "alagapan_stimulation_geometry.json"
    haslacher_path = RESULTS / "haslacher_stimulation_geometry.json"
    alagapan = json.loads(alagapan_path.read_text()) if alagapan_path.exists() else {}
    haslacher = json.loads(haslacher_path.read_text()) if haslacher_path.exists() else {}
    return {
        "question": "config key 'alagapan_phase_stimulation' names 'phase stimulation'; is it the "
                    "non-invasive phase-tuned arm this leg's third arm requires?",
        "finding": "no",
        "evidence": {
            "alagapan_stimulation_geometry.json": {
                "n_patients": alagapan.get("_meta", {}).get("n_patients"),
                "citation": alagapan.get("_meta", {}).get("citation"),
                "condition_labels_seen": ["In Phase", "Anti Phase", "Sham"] if "P1" in alagapan else None,
                "modality": "invasive intracranial depth electrodes",
            },
            "haslacher_stimulation_geometry.json": {
                "n_active": haslacher.get("active", {}).get("n_subjects"),
                "n_control": haslacher.get("control", {}).get("n_subjects"),
                "citation": haslacher.get("_meta", {}).get("citation"),
                "modality": "non-invasive scalp EEG / transcranial alternating-current stimulation",
                "group_labels_seen": ["active", "control"] if "active" in haslacher else None,
            },
        },
        "conclusion": ("'alagapan_phase_stimulation' is an invasive, three-patient, intracranial "
                       "encoding-stimulation dataset whose read-out is a retention-period aftereffect "
                       "(condition labels In Phase / Anti Phase / Sham within each of 3 patients). It is "
                       "NOT this leg's third arm and is excluded from the four-arm map; it is re-run here "
                       "only so its own delivered reproduction can be checked. This leg's third arm is the "
                       "non-invasive scalp dataset with a designed active-group-versus-control-group-"
                       "stimulated-away-from-source assignment (occipital active electrodes vs. frontal "
                       "control electrodes), a phase-tuned alternating-current stimulation, and a "
                       "working-memory retention-period read-out."),
    }


# ── Confound table ───────────────────────────────────────────────────────────────

def confound_table(n_by_arm: dict, arm_void_reason: dict) -> list[dict]:
    """arm_void_reason[arm] is None when the arm was actually computed this run
    (n_by_arm[arm] is then a real session/participant count) or a disclosed
    reason string when this run's own reproduction gate voided the arm before
    it was computed (n_by_arm[arm] is then None -- not a measured zero). Both
    dicts are built with the fixed 4-arm key set in main() below, so a direct
    subscript here raises if that ever stops being true rather than silently
    reporting an absent arm's null as if it had been measured.
    """
    def _n_status(arm: str) -> dict:
        reason = arm_void_reason[arm]
        return ({"n_status": "computed"} if reason is None else
                {"n_status": "not_computed_this_run_reproduction_gate_void", "n_status_reason": reason})

    return [
        {
            "arm": "ds005489_openloop", "species": "human", "invasive_or_noninvasive": "invasive",
            "stimulation_waveform": "intracranial electrical stimulation, item/pair-randomized within stim lists",
            "stimulated_epoch": "episodic encoding (word presentation)",
            "readout_epoch": "same encoding-window neural trajectory",
            "task": "delayed free recall (word list)",
            "assignment_mechanism": "randomized (item/pair-level, within stim lists)",
            "unit_of_analysis": "session", "n": n_by_arm["ds005489_openloop"], **_n_status("ds005489_openloop"),
            "delivered_causal_gate_status": "G3 causal (delivered; not re-derived here)",
        },
        {
            "arm": "ds005557_closedloop", "species": "human", "invasive_or_noninvasive": "invasive",
            "stimulation_waveform": "intracranial electrical stimulation, classifier-triggered within stim lists",
            "stimulated_epoch": "episodic encoding (word presentation)",
            "readout_epoch": "same encoding-window neural trajectory",
            "task": "delayed free recall (word list)",
            "assignment_mechanism": "list-level randomized; item-level classifier-triggered "
                                    "(propensity-selected on the measured neural state)",
            "unit_of_analysis": "session", "n": n_by_arm["ds005557_closedloop"], **_n_status("ds005557_closedloop"),
            "delivered_causal_gate_status": "list-level: G3 causal; item-level: descriptive only, "
                                            "never causal (delivered restriction, carried forward unchanged)",
        },
        {
            "arm": "haslacher_clam_tacs", "species": "human", "invasive_or_noninvasive": "non-invasive",
            "stimulation_waveform": "phase-tuned transcranial alternating-current stimulation (tACS), "
                                    "individually tuned to each participant's own retention-period alpha phase",
            "stimulated_epoch": "continuous during task blocks, phase-locked in real time",
            "readout_epoch": "working-memory retention period (dataset-defined window)",
            "task": "visual working memory with phase-locked tACS enhancement",
            "assignment_mechanism": "designed group assignment (active occipital electrodes vs. control "
                                    "frontal electrodes stimulated away from the source), not trial-randomized",
            "unit_of_analysis": "participant", "n": n_by_arm["haslacher_clam_tacs"], **_n_status("haslacher_clam_tacs"),
            "delivered_causal_gate_status": "no causal gate delivered for this arm; reported descriptively here",
        },
        {
            "arm": "macaque_pfc_microstimulation", "species": "macaque", "invasive_or_noninvasive": "invasive",
            "stimulation_waveform": "intracortical microstimulation via the recording electrode array(s)",
            "stimulated_epoch": "working-memory delay period",
            "readout_epoch": "same delay-period spike-rate trajectory, stimulation-onset-aligned",
            "task": "memory-guided reach to a remembered target angle",
            "assignment_mechanism": "designed (near-known) propensity, experimenter-set stimulation-condition sequence",
            "unit_of_analysis": "session", "n": n_by_arm["macaque_pfc_microstimulation"], **_n_status("macaque_pfc_microstimulation"),
            "delivered_causal_gate_status": "FAIL/WEAK (delivered; carried forward unchanged, not re-run)",
        },
    ]


# ── Cross-arm assembly ────────────────────────────────────────────────────────────

def ram_arm_pooled(arm_result: dict, key: str) -> dict:
    """key: 'normalized_displacement' path for openloop, or a lambda-free
    accessor name for closedloop sub-blocks -- both call sites build the
    per-session dict explicitly, this just wraps pool_sessions."""
    values = {}
    for session_id, rec in arm_result["per_session"].items():
        v = rec
        for part in key.split("."):
            v = v.get(part) if isinstance(v, dict) else None
            if v is None:
                break
        if v is not None:
            values[session_id] = v
    return pool_sessions(values, DEFAULT_N_NULL_DRAWS)


def macaque_pfc_microstimulation_arm_pooled(arm_result: dict) -> dict:
    values = {}
    for session_id, rec in arm_result["per_session"].items():
        conds = rec.get("conditions", {})
        norm_vals = [c["normalized_displacement"] for c in conds.values() if c.get("normalized_displacement") is not None]
        if norm_vals:
            values[session_id] = float(np.mean(norm_vals))
    return pool_sessions(values, DEFAULT_N_NULL_DRAWS)


def haslacher_arm_pooled(arm_result: dict) -> dict:
    values = {sid: rec["normalized_displacement"] for sid, rec in arm_result["per_subject"].items()
              if rec.get("normalized_displacement") is not None}
    return pool_sessions(values, DEFAULT_N_NULL_DRAWS)


def spontaneous_sd_comparability(raw_sds: dict, backing_n: dict) -> dict:
    """normalisation_does_not_make_the_arms_comparable: is the spontaneous-
    variability denominator itself estimable everywhere, and is it
    well-determined everywhere.

    The raw magnitude of the denominator is NOT itself evidence of
    incomparability: a microvolt EEG amplitude, a spikes/second rate and a
    z-scored PC-score deviation will always differ by orders of magnitude in
    raw units by construction, regardless of whether the ratio each arm's
    own denominator produces is trustworthy -- that units mismatch is
    exactly the problem this leg's normalisation exists to remove, not
    evidence it failed. What WOULD make the ratio uninterpretable is the
    denominator itself being too poorly determined to divide by: its
    relative standard error, from the classical sd-of-a-sample-sd
    approximation SE(sd)/sd ~= 1/sqrt(2*(n-1)) for the median unit's own
    backing trial count `n`, compared across arms.
    """
    estimable = {arm: v for arm, v in raw_sds.items() if v is not None and v > 0}
    inestimable_arms = [arm for arm, v in raw_sds.items() if v is None or not (v > 0)]
    relative_se = {}
    for arm, n in backing_n.items():
        if n is not None and n > 1:
            relative_se[arm] = float(1.0 / np.sqrt(2.0 * (n - 1)))
    poorly_determined_arms = [arm for arm, se in relative_se.items() if se > 0.5]
    if len(estimable) < 2:
        return {"status": "not_evaluable", "reason": "fewer than 2 arms with an estimable spontaneous sd",
                "inestimable_arms": inestimable_arms, "raw_spontaneous_sd_by_arm": raw_sds}
    values = np.array(list(estimable.values()))
    return {
        "status": "evaluated", "raw_spontaneous_sd_by_arm": raw_sds,
        "raw_units_differ_by_construction": "raw units are not commensurable across arms (see confound_table "
                                            "modality); the span below is expected and is NOT used to judge "
                                            "comparability -- see relative_se_of_spontaneous_sd_by_arm instead",
        "min_sd": float(values.min()), "max_sd": float(values.max()),
        "span_orders_of_magnitude": float(np.log10(values.max() / values.min())),
        "inestimable_arms": inestimable_arms,
        "median_backing_trial_count_by_arm": backing_n,
        "relative_se_of_spontaneous_sd_by_arm": relative_se,
        "poorly_determined_arms": poorly_determined_arms,
        "not_comparable": bool(inestimable_arms or poorly_determined_arms),
    }


def determine_named_outcomes(arms: dict, comparability: dict) -> dict:
    per_arm_significance = {}
    for arm_name, pooled in arms.items():
        if pooled.get("status") != "complete":
            per_arm_significance[arm_name] = {"status": pooled.get("status", "not_estimable"),
                                              "reason": pooled.get("reason")}
            continue
        mdd = pooled.get("mdd_80pct_power")
        below_floor = bool(pooled.get("significant") is False and mdd is not None
                            and abs(pooled["mean_diff"]) < mdd)
        per_arm_significance[arm_name] = {
            "status": "evaluated", "significant": pooled["significant"], "mean_diff": pooled["mean_diff"],
            "ci_lower": pooled["ci_lower"], "ci_upper": pooled["ci_upper"], "p_value": pooled["p_value"],
            "sd": pooled.get("sd"), "mdd_80pct_power": mdd, "n": pooled["n"],
            "inconclusive_below_detection_floor": below_floor,
        }

    n_total_arms = len(arms)
    evaluated = {a: r for a, r in per_arm_significance.items() if r["status"] == "evaluated"}
    not_evaluated = {a: r["status"] for a, r in per_arm_significance.items() if r["status"] != "evaluated"}
    n_significant = sum(1 for r in evaluated.values() if r["significant"])
    n_evaluated = len(evaluated)

    outcomes = {
        "per_arm_significance": per_arm_significance,
        "n_total_arms": n_total_arms, "n_evaluated_arms": n_evaluated, "n_significant_arms": n_significant,
        "arms_not_evaluated": not_evaluated,
    }
    if comparability.get("status") == "evaluated" and comparability.get("not_comparable"):
        outcomes["fired"] = "normalisation_does_not_make_the_arms_comparable"
        outcomes["comparability_detail"] = comparability
    elif n_evaluated == 0:
        outcomes["fired"] = "gap_not_covered_by_the_named_outcome_list"
        outcomes["gap_note"] = "no arm produced an evaluable pooled displacement statistic"
    elif n_evaluated < n_total_arms:
        # "every arm" requires all four; "some arms only" requires multiple
        # arms to have actually been contrasted. Neither is true when most
        # arms never reached evaluation (a reproduction-gate failure is not
        # a null result) -- forcing either label here would overstate what
        # was actually measured, so this is reported as the gap it is.
        outcomes["fired"] = "gap_not_covered_by_the_named_outcome_list"
        outcomes["gap_note"] = (
            f"only {n_evaluated} of {n_total_arms} arms reached an evaluable pooled displacement statistic "
            f"({', '.join(sorted(not_evaluated)) if not_evaluated else 'none'} did not, see arms_not_evaluated); "
            "the named outcome list's 'every arm' and 'some arms only' both presume multiple arms were "
            "actually evaluated and contrasted, which did not happen here.")
        if n_evaluated > 0:
            outcomes["evaluated_arms_all_significant"] = bool(n_significant == n_evaluated)
        if n_evaluated == 1 and n_significant == 1:
            outcomes["superseded_outcome"] = {
                "previously_fired": "stimulation_displaces_the_latent_state_in_every_arm",
                "why_that_was_wrong": (
                    "a bug in this artifact's own outcome-selection logic counted 'every arm significant' "
                    "against the number of EVALUATED arms rather than the fixed total of four arms, so a "
                    "single evaluated arm being significant was read as unanimous agreement across all "
                    "four; the other three arms were void on their own reproduction gate at that time, not "
                    "absent from the count. Corrected here to compare against the fixed arm count."),
            }
    elif n_significant == n_evaluated:  # implies n_evaluated == n_total_arms, reached only via the branch above
        outcomes["fired"] = "stimulation_displaces_the_latent_state_in_every_arm"
    elif n_significant > 0:
        outcomes["fired"] = "stimulation_displaces_the_latent_state_in_some_arms_only"
        outcomes["note"] = ("Significant in some arms and not others. The four arms differ in species, "
                            "invasiveness, stimulation waveform, stimulated epoch, read-out epoch, task and "
                            "assignment mechanism simultaneously (see confound_table); no explanation is "
                            "asserted for which arms differ.")
    else:
        any_below_floor = any(r["inconclusive_below_detection_floor"] for r in evaluated.values())
        if any_below_floor:
            outcomes["fired"] = "inconclusive_below_detection_floor"
            outcomes["reference_effects"] = {a: r["mdd_80pct_power"] for a, r in evaluated.items()}
        else:
            outcomes["fired"] = "gap_not_covered_by_the_named_outcome_list"
            outcomes["gap_note"] = ("no arm shows a significant displacement, but not every non-significant "
                                    "arm's minimum detectable difference exceeds its observed effect -- the "
                                    "named outcome list does not cover this combination; numbers are in "
                                    "per_arm_significance above")
    return outcomes


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openloop-max-subjects", type=int, default=38)
    p.add_argument("--closedloop-max-subjects", type=int, default=35)
    p.add_argument("--macaque_pfc_microstimulation-max-sessions", type=int, default=11)
    p.add_argument("--haslacher-max-active", type=int, default=21)
    p.add_argument("--haslacher-max-control", type=int, default=25)
    p.add_argument("--n-null-draws", type=int, default=DEFAULT_N_NULL_DRAWS)
    p.add_argument("--n-rotation-null", type=int, default=DEFAULT_N_ROTATION_NULL)
    p.add_argument("--n-jobs", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--skip-reproduction-gate", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    t0 = time.time()

    if args.skip_reproduction_gate:
        reproduction_gate = {arm: {"status": "skipped_by_flag"} for arm in
                             ("ram_stimulation_drift", "causal_macaque_pfc_microstimulation", "alagapan_stimulation_geometry",
                              "haslacher_stimulation_geometry")}
    else:
        reproduction_gate = {
            "ram_stimulation_drift": run_checkpointed(
                "reproduction__ram_stimulation_drift",
                lambda: reproduce_arm("run_ram_stimulation_drift", "ram_stimulation_drift")),
            "causal_macaque_pfc_microstimulation": run_checkpointed(
                "reproduction__causal_macaque_pfc_microstimulation",
                lambda: reproduce_arm("run_macaque_pfc_microstimulation_pipeline", "causal_macaque_pfc_microstimulation")),
            "alagapan_stimulation_geometry": run_checkpointed(
                "reproduction__alagapan_stimulation_geometry",
                lambda: reproduce_arm("run_alagapan_stimulation_geometry", "alagapan_stimulation_geometry")),
            "haslacher_stimulation_geometry": run_checkpointed(
                "reproduction__haslacher_stimulation_geometry",
                lambda: reproduce_arm("run_haslacher_stimulation_geometry", "haslacher_stimulation_geometry",
                                      extra_inputs=["haslacher_phase_omega.json"])),
        }
    print(f"[reproduction gate] {json.dumps({k: v['status'] for k, v in reproduction_gate.items()})}", flush=True)

    arm3_note = disambiguate_arm3()
    print(f"[arm 3 disambiguation] {arm3_note['conclusion'][:120]}...", flush=True)

    from run_ram_stimulation_drift import N_PC as RAM_N_PC

    openloop_void = reproduction_gate["ram_stimulation_drift"]["status"] == "void_reproduction_gate_did_not_reproduce"
    if openloop_void:
        openloop = {"status": "void_reproduction_gate_did_not_reproduce",
                    "reason": reproduction_gate["ram_stimulation_drift"].get("reason")}
        closedloop = dict(openloop)
    else:
        t1 = time.time()
        openloop = ram_arm("ram_ds005489_openloop", args.openloop_max_subjects, False, ram_openloop_session,
                           "ram_openloop", RAM_N_PC, args.n_null_draws, args.n_jobs)
        print(f"[ram openloop] {len(openloop['per_session'])} sessions computed in {time.time()-t1:.0f}s", flush=True)
        t1 = time.time()
        closedloop = ram_arm("ram_ds005557_closedloop", args.closedloop_max_subjects, True, ram_closedloop_session,
                             "ram_closedloop", RAM_N_PC, args.n_null_draws, args.n_jobs)
        print(f"[ram closedloop] {len(closedloop['per_session'])} sessions computed in {time.time()-t1:.0f}s", flush=True)

    macaque_pfc_microstimulation_void = reproduction_gate["causal_macaque_pfc_microstimulation"]["status"] == "void_reproduction_gate_did_not_reproduce"
    if macaque_pfc_microstimulation_void:
        macaque_pfc_microstimulation = {"status": "void_reproduction_gate_did_not_reproduce",
                  "reason": reproduction_gate["causal_macaque_pfc_microstimulation"].get("reason")}
    else:
        t1 = time.time()
        macaque_pfc_microstimulation = macaque_pfc_microstimulation_arm(args.macaque_pfc_microstimulation_max_sessions, 8, args.n_null_draws, args.n_rotation_null)
        print(f"[macaque_pfc_microstimulation] {len(macaque_pfc_microstimulation['per_session'])} sessions computed in {time.time()-t1:.0f}s", flush=True)

    haslacher_void = reproduction_gate["haslacher_stimulation_geometry"]["status"] == "void_reproduction_gate_did_not_reproduce"
    if haslacher_void:
        haslacher = {"status": "void_reproduction_gate_did_not_reproduce",
                     "reason": reproduction_gate["haslacher_stimulation_geometry"].get("reason")}
    else:
        t1 = time.time()
        haslacher = haslacher_arm(args.haslacher_max_active, args.haslacher_max_control, "parallel_analysis",
                                  args.n_null_draws, args.n_rotation_null)
        print(f"[haslacher] {len(haslacher['per_subject'])} participants computed in {time.time()-t1:.0f}s", flush=True)

    n_by_arm = {
        "ds005489_openloop": len(openloop.get("per_session", {})) if not openloop_void else None,
        "ds005557_closedloop": len(closedloop.get("per_session", {})) if not openloop_void else None,
        "haslacher_clam_tacs": len(haslacher.get("per_subject", {})) if not haslacher_void else None,
        "macaque_pfc_microstimulation": len(macaque_pfc_microstimulation.get("per_session", {})) if not macaque_pfc_microstimulation_void else None,
    }
    # None above always means "this run's own reproduction gate voided the arm before it was
    # computed" -- never a measured zero. Carried alongside n_by_arm so confound_table can record
    # that distinction explicitly instead of writing an unqualified null into the delivered artifact.
    arm_void_reason = {
        "ds005489_openloop": openloop.get("reason") if openloop_void else None,
        "ds005557_closedloop": closedloop.get("reason") if openloop_void else None,
        "haslacher_clam_tacs": haslacher.get("reason") if haslacher_void else None,
        "macaque_pfc_microstimulation": macaque_pfc_microstimulation.get("reason") if macaque_pfc_microstimulation_void else None,
    }

    pooled = {}
    pooled["ds005489_openloop"] = ({"status": "void", "reason": openloop.get("reason")} if openloop_void
                                    else ram_arm_pooled(openloop, "displacement.normalized_displacement"))
    pooled["ds005557_closedloop_list_level"] = ({"status": "void", "reason": openloop_void and closedloop.get("reason")}
                                                if openloop_void else
                                                ram_arm_pooled(closedloop, "list_level.displacement.normalized_displacement"))
    pooled["ds005557_closedloop_item_level_descriptive"] = (
        {"status": "void", "reason": closedloop.get("reason")} if openloop_void else
        ram_arm_pooled(closedloop, "item_level.displacement.normalized_displacement"))
    pooled["haslacher_clam_tacs"] = ({"status": "void", "reason": haslacher.get("reason")} if haslacher_void
                                     else haslacher_arm_pooled(haslacher))
    pooled["macaque_pfc_microstimulation"] = ({"status": "void", "reason": macaque_pfc_microstimulation.get("reason")} if macaque_pfc_microstimulation_void
                               else macaque_pfc_microstimulation_arm_pooled(macaque_pfc_microstimulation))

    headline_arms = {
        "ds005489_openloop": pooled["ds005489_openloop"],
        "ds005557_closedloop": pooled["ds005557_closedloop_list_level"],
        "haslacher_clam_tacs": pooled["haslacher_clam_tacs"],
        "macaque_pfc_microstimulation": pooled["macaque_pfc_microstimulation"],
    }

    def _arm_median(arm_result, key, per_session_key):
        if isinstance(arm_result, dict) and arm_result.get("status") == "void_reproduction_gate_did_not_reproduce":
            return None
        vals = []
        for rec in arm_result.get(per_session_key, {}).values():
            v = rec
            for part in key.split("."):
                v = v.get(part) if isinstance(v, dict) else None
                if v is None:
                    break
            if v is not None:
                vals.append(v)
        return float(np.median(vals)) if vals else None

    raw_sds = {
        "ds005489_openloop": _arm_median(openloop, "displacement.spontaneous_sd_raw", "per_session"),
        "ds005557_closedloop": _arm_median(closedloop, "list_level.displacement.spontaneous_sd_raw", "per_session"),
        "haslacher_clam_tacs": _arm_median(haslacher, "spontaneous_sd_raw", "per_subject"),
        "macaque_pfc_microstimulation": _arm_median(macaque_pfc_microstimulation, "spontaneous_sd_raw", "per_session"),
    }
    backing_n = {
        "ds005489_openloop": _arm_median(openloop, "displacement.n_unstim_trials", "per_session"),
        "ds005557_closedloop": _arm_median(closedloop, "list_level.displacement.n_unstim_trials", "per_session"),
        "haslacher_clam_tacs": _arm_median(haslacher, "n_baseline_trials", "per_subject"),
        "macaque_pfc_microstimulation": _arm_median(macaque_pfc_microstimulation, "n_control_trials_for_frame", "per_session"),
    }
    comparability = spontaneous_sd_comparability(raw_sds, backing_n)
    named_outcomes = determine_named_outcomes(headline_arms, comparability)

    artifact = {
        "schema_version": "1.0.0", "analysis_id": "stimulation_latent_response_map",
        "code_commit": git_commit(ROOT), "runtime_seconds": time.time() - t0,
        "mandatory_conditions": {"n_null_draws": args.n_null_draws, "n_rotation_null_draws": args.n_rotation_null,
                                 "unstimulated_trial_null": "stimulation label permuted within session or "
                                 "participant, whole displacement statistic (frame refit through normalisation) "
                                 "recomputed end to end per draw"},
        "reproduction_gate": reproduction_gate,
        "reproduction_gate_investigation": reproduction_gate_investigation(reproduction_gate, ROOT),
        "arm_3_disambiguation": arm3_note,
        "confound_table": confound_table(n_by_arm, arm_void_reason),
        "cross_arm_comparability": comparability,
        "arms": {
            "ds005489_openloop": openloop,
            "ds005557_closedloop": closedloop,
            "haslacher_clam_tacs": haslacher,
            "macaque_pfc_microstimulation": macaque_pfc_microstimulation,
        },
        "pooled_normalized_displacement": pooled,
        "named_outcomes": named_outcomes,
        "limitations": [
            "This is not an epoch, species, or modality contrast: every arm differs from every other arm in "
            "species, recording modality, task, stimulation waveform, stimulation site and stimulated epoch "
            "simultaneously. No cross-arm difference in this artifact is attributed to any one of those "
            "factors; see confound_table before reading any cross-arm comparison.",
            "The macaque delay-period causal-gate verdict (macaque_pfc_microstimulation) is FAIL/WEAK and is carried forward "
            "unchanged in this artifact -- it is not re-run and its status is never upgraded.",
            "ds005557_closedloop's item-level (classifier-triggered) comparison is descriptive only and never "
            "causal: assignment is propensity-selected on the very neural state this analysis measures.",
            "ds005489_openloop and ds005557_closedloop keep the delivered arm's own session-specific "
            "unsupervised leading component (PC1 only) as their latent frame; no multi-dimensional dynamics "
            "model or supervised content label exists in that delivered pipeline, so content-subspace "
            "alignment and control-relevant-direction alignment are not computable for those two arms and are "
            "reported as such rather than forced onto a nearby number.",
            "haslacher_clam_tacs has no supervised memory-content label in its delivered pipeline; the "
            "content-subspace alignment result reports that by name and substitutes alignment to the arm's own "
            "unsupervised leading dynamics mode (v_star) instead.",
            "Naming note: the macaque_pfc_microstimulation arm identifier previously named the "
            "data-releasing laboratory instead of describing the preparation; it has been renamed to a "
            "purely descriptive label, and its checkpoint filenames and reproduction-gate keys moved with "
            "it. No checkpoint reader validates a stored key or filename against the one requested, so the "
            "move preserved every already-computed fit rather than discarding it. The haslacher_clam_tacs "
            "arm identifier still names the data-releasing laboratory and remains unrenamed; that corpus "
            "was out of scope for this fix.",
        ],
    }

    destination = RESULTS / "stimulation_latent_response_map.json"
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(canonical_json(artifact))
    os.replace(tmp, destination)
    print(f"\nSaved {destination} in {time.time()-t0:.0f}s total", flush=True)
    print(f"Named outcome fired: {named_outcomes['fired']}", flush=True)


if __name__ == "__main__":
    main()
